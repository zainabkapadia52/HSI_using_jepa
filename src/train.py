import os
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('SLURM_LOCALID', '')
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch

from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms
from src.datasets.houston13 import make_houston13

# ------------------- SETTINGS -------------------
log_timings = True
log_freq = 10
checkpoint_freq = 50
_GLOBAL_SEED = 0

torch.manual_seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):
    # -------------------- META --------------------
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name   = args['meta']['model_name']
    load_model   = args['meta']['load_checkpoint'] or resume_preempt
    r_file       = args['meta']['read_checkpoint']
    pred_depth   = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)

    # -------------------- DATA --------------------
    batch_size = args['data']['batch_size']
    # Maintain original effective batch size of 128 via gradient accumulation
    effective_bs = 128
    grad_accum_steps = max(1, effective_bs // batch_size)

    pin_mem     = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    crop_size   = args['data']['crop_size']
    crop_scale  = args['data']['crop_scale']
    image_folder= args['data']['image_folder']
    root_path   = args['data']['root_path']

    # -------------------- MASK --------------------
    mask_params = args['mask']

    # ---------------- OPTIMIZATION ----------------
    opt_params = args['optimization']

    # ----------------- LOGGING --------------------
    folder = args['logging']['folder']
    tag    = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'params-ijepa.yaml'), 'w') as f:
        yaml.dump(args, f)

    # ------------- DISTRIBUTED INIT ----------------
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # --------------- CSV LOGGER -------------------
    log_file    = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    csv_logger = CSVLogger(log_file,
                           ('%d','epoch'),
                           ('%d','itr'),
                           ('%.5f','loss'),
                           ('%.5f','mask-A'),
                           ('%.5f','mask-B'),
                           ('%d','time (ms)'))

    # ---------------- MODEL ----------------------
    encoder, predictor = init_model(
        device=device,
        patch_size=mask_params['patch_size'],
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    # override patch embed for 144 channels
    in_ch = 144
    proj = encoder.patch_embed.proj
    encoder.patch_embed.proj = torch.nn.Conv2d(
        in_channels=in_ch,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=(proj.bias is not None)
    )
    torch.nn.init.kaiming_uniform_(encoder.patch_embed.proj.weight, nonlinearity='relu')
    if proj.bias is not None:
        torch.nn.init.constant_(encoder.patch_embed.proj.bias, 0)

    target_encoder = copy.deepcopy(encoder)
    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    if world_size>1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters(): p.requires_grad = False

    # ------------ TRANSFORMS & DATA ------------
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=mask_params['patch_size'],
        pred_mask_scale=mask_params['pred_mask_scale'],
        enc_mask_scale=mask_params['enc_mask_scale'],
        aspect_ratio=mask_params['aspect_ratio'],
        nenc=mask_params['num_enc_masks'],
        npred=mask_params['num_pred_masks'],
        allow_overlap=mask_params['allow_overlap'],
        min_keep=mask_params['min_keep'])

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        horizontal_flip=args['data']['use_horizontal_flip'],
        normalize_mean=args['transforms']['normalize_mean'],
        normalize_std=args['transforms']['normalize_std'],
        band_dropout=args['transforms']['band_dropout'],
        spectral_noise=args['transforms']['spectral_noise'])

    _, loader, sampler = make_houston13(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        drop_last=True)
    iterations_per_epoch = len(loader)

    # ---------------- OPTIMIZER -------------------
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=float(opt_params['weight_decay']),
        final_wd=float(opt_params['final_weight_decay']),
        start_lr=opt_params['start_lr'],
        ref_lr=opt_params['lr'],
        final_lr=opt_params['final_lr'],
        iterations_per_epoch=iterations_per_epoch,
        warmup=opt_params['warmup'],
        num_epochs=opt_params['epochs'],
        ipe_scale=opt_params['ipe_scale'],
        use_bfloat16=use_bfloat16)
    
    # ———— resume from checkpoint if requested ————
    start_epoch = 0
    if load_model and r_file:
        if os.path.isfile(latest_path):
            (encoder, predictor, target_encoder,
            optimizer, scaler, start_epoch) = load_checkpoint(
                device,
                latest_path,
                encoder if world_size==1 else encoder.module,
                predictor if world_size==1 else predictor.module,
                target_encoder if world_size==1 else target_encoder.module,
                optimizer,
                scaler
            )
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {latest_path}, starting from scratch.")


    momentum_schedule = (
        opt_params['ema'][0] + i*(opt_params['ema'][1]-opt_params['ema'][0])
        /(iterations_per_epoch*opt_params['epochs']*opt_params['ipe_scale'])
        for i in range(iterations_per_epoch * opt_params['epochs'] + 1)
    )

    optimizer.zero_grad()


    # --------------- TRAIN LOOP -------------------
    for epoch in range(start_epoch, opt_params['epochs']):
        sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(loader):
            imgs = udata[0].to(device, non_blocking=True)
            masks_enc  = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                with torch.no_grad():  # target
                    h = target_encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    B = h.size(0)
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                # context
                z = encoder(imgs, masks_enc)
                z = predictor(z, masks_enc, masks_pred)
                loss = F.smooth_l1_loss(z, h)
                loss = AllReduce.apply(loss)

            # gradient accumulation
            loss = loss / grad_accum_steps
            if use_bfloat16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update stats
            loss_meter.update(float(loss) * grad_accum_steps)
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            # step
            if (itr + 1) % grad_accum_steps == 0:
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                if use_bfloat16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # momentum
                with torch.no_grad():
                    m = next(momentum_schedule)
                    for q, k in zip(encoder.parameters(), target_encoder.parameters()):
                        k.data.mul_(m).add_((1. - m) * q.detach().data)
                optimizer.zero_grad()

            # logging
            if itr % log_freq == 0:
                logger.info(f"[Epoch {epoch+1}, itr {itr}] loss: {loss_meter.avg:.3f}")
                csv_logger.log(epoch+1, itr, loss_meter.avg, maskA_meter.val, maskB_meter.val, time_meter.avg)

        # checkpoint
        save_dict = {
            'encoder':         encoder.state_dict(),
            'predictor':       predictor.state_dict(),
            'opt':             optimizer.state_dict(),
            'epoch':           epoch+1,
            'target_encoder':  target_encoder.state_dict(),
        }
        if scaler is not None:
            save_dict['scaler'] = scaler.state_dict()

        torch.save(save_dict, latest_path)



if __name__ == '__main__':
    cfg = yaml.safe_load(open(sys.argv[1], 'r'))
    main(cfg)
