#!/usr/bin/env python3
# Smoke-test for Houston-13 IJepa pipeline (full encoder+predictor)

import os, sys
# Ensure project root is on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from logging import getLogger
from src.utils.distributed import init_distributed
from src.masks.multiblock import MaskCollator
from src.helper import init_model
from src.utils.tensors import repeat_interleave_batch
from src.masks.utils import apply_masks
from src.transforms import make_transforms
from src.datasets.houston13 import make_houston13

logger = getLogger()

print("Starting HSI IJepa full pipeline test...")

# 1) Distributed setup (single-GPU)
world_size, rank = init_distributed(rank_and_world_size=(0, 1))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2) Build transforms (no jitter)
transform = make_transforms(
    crop_size=19,
    crop_scale=(1.0, 1.0),
    horizontal_flip=False,
    normalize_mean=None,
    normalize_std=None,
    band_dropout=0,
    spectral_noise=0.0
)

# 3) Mask collator (1 context, 1 pred block)
mask_collator = MaskCollator(
    input_size=19,
    patch_size=3,
    enc_mask_scale=(0.1, 0.1),
    pred_mask_scale=(0.1, 0.1),
    aspect_ratio=(1.0, 1.0),
    nenc=1,
    npred=1,
    allow_overlap=False,
    min_keep=1
)

# 4) DataLoader
dataset, loader, sampler = make_houston13(
    transform=transform,
    batch_size=1,
    collator=mask_collator,
    pin_mem=False,
    num_workers=0,
    world_size=world_size,
    rank=rank,
    root_path='/Users/zainab/HSI_using_jepa/dataset',
    image_folder='Houston13_patches',
    drop_last=True
)

# 5) Model init
encoder, predictor = init_model(
    device=device,
    patch_size=3,
    crop_size=19,
    pred_depth=1,
    pred_emb_dim=96,   # must be divisible by num_heads (3)
    model_name='vit_tiny'
)

# Override patch-embed to match dataset channels
in_ch = dataset[0].shape[0]
print(f"Overriding patch_embed to in_channels={in_ch}")
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

def test_one_batch():
    print("Entering test_one_batch...")
    sampler.set_epoch(0)
    for udata, masks_enc, masks_pred in loader:
        print("Fetched batch from DataLoader")
        print("udata shape:", udata.shape)

        imgs = udata.to(device)
        print("imgs on device, shape:", imgs.shape)

        # Context branch
        z_enc = encoder(imgs, masks_enc)
        print("Context encoder output shape:", z_enc.shape)

        # Predictor branch
        z_pred = predictor(z_enc, masks_enc, masks_pred)
        print("Predictor output shape:", z_pred.shape)

        # Target branch (ground-truth tokens)
        h = encoder(imgs)
        B = imgs.size(0)
        h_norm = F.layer_norm(h, (h.size(-1),))
        h_masked = apply_masks(h_norm, masks_pred)
        h_rep = repeat_interleave_batch(h_masked, B, repeat=len(masks_enc))
        print("Target output shape:", h_rep.shape)

        print("Full pipeline smoke test successful!")
        break

if __name__ == '__main__':
    test_one_batch()
