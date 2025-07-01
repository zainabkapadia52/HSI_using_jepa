#!/usr/bin/env python3
# Sanity-check training loop for Houston-13 IJepa pipeline
# Runs a single gradient step on one batch to verify training integration.

import os, sys
from logging import getLogger

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, distributed
from src.utils.distributed import init_distributed
from src.masks.multiblock import MaskCollator
from src.helper import init_model
from src.transforms import make_transforms
from src.datasets.houston13 import make_houston13
from src.utils.tensors import repeat_interleave_batch
from src.masks.utils import apply_masks

logger = getLogger()

def main():
    # Distributed setup (1 GPU)
    world_size, rank = init_distributed(rank_and_world_size=(0, 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = make_transforms(
        crop_size=19,
        crop_scale=(0.8, 1.0),
        horizontal_flip=True,
        normalize_mean=None,
        normalize_std=None,
        band_dropout=8,
        spectral_noise=0.01
    )

    # Mask collator (loosened for small patch grid)
    mask_collator = MaskCollator(
        input_size=19,
        patch_size=3,
        enc_mask_scale=(0.15, 0.3),    # larger context blocks
        pred_mask_scale=(0.1, 0.2),    # larger target blocks
        aspect_ratio=(0.75, 1.5),
        nenc=1,                         # fewer context masks
        npred=1,                        # fewer target masks
        allow_overlap=False,
        min_keep=1                      # minimum tokens to keep
    )

    # DataLoader
    dataset, loader, sampler = make_houston13(
        transform=transform,
        batch_size=4,
        collator=mask_collator,
        pin_mem=True,
        num_workers=0,  # use single-process loader to avoid pickle issues
        world_size=world_size,
        rank=rank,
        root_path='/Users/zainab/HSI_using_jepa/dataset',
        image_folder='Houston13_patches',
        drop_last=True
    )

    # Model init
    encoder, predictor = init_model(
        device=device,
        patch_size=3,
        crop_size=19,
        pred_depth=36,
        pred_emb_dim=384,
        model_name='vit_huge'
    )

    # Override patch_embed in_channels to match hyperspectral bands
    in_ch = dataset[0].shape[0]
    proj = encoder.patch_embed.proj
    encoder.patch_embed.proj = torch.nn.Conv2d(
        in_channels=in_ch,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=(proj.bias is not None)
    )
    encoder.to(device)
    predictor.to(device)

    # Optimizer
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    # Single batch training step
    sampler.set_epoch(0)
    for udata, masks_enc, masks_pred in loader:
        imgs = udata.to(device)

        # Encoder forward
        z_enc = encoder(imgs, masks_enc)
        # Predictor forward
        z_pred = predictor(z_enc, masks_enc, masks_pred)

        # Ground-truth target embeddings
        h = encoder(imgs)
        B = imgs.size(0)
        h_norm = F.layer_norm(h, (h.size(-1),))
        h_masked = apply_masks(h_norm, masks_pred)
        h_target = repeat_interleave_batch(h_masked, B, repeat=len(masks_enc))

        # Compute reconstruction loss
        loss = F.mse_loss(z_pred, h_target)
        print(f"Sanity train step loss: {loss.item():.4f}")

        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

if __name__ == '__main__':
    print("Running one training step for sanity-check...")
    main()
