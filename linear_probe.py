#!/usr/bin/env python3
"""
Complete script for running a linear probe on the Houston HSI dataset
using your pretrained IJ-EPA encoder. It reads your training YAML config,
loads the .mat file, extracts 3×3 patches of hyperspectral data,
passes them through the frozen encoder to get features, and trains
a multinomial logistic regression classifier.

Debug prints added to verify patch extraction, batch shapes, and
that you're actually running this script, and that the patch_embed conv
was correctly overridden.

Usage:
  python linear_probe.py \
      --config path/to/ijepa_config.yaml \
      --mat    path/to/houston_data.mat \
      --ckpt   path/to/latest.pth.tar
"""
import os
import argparse
import yaml
import numpy as np
import scipy.io
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.helper import init_model

# Confirm which file is running
print(f"DEBUG: running script at {os.path.realpath(__file__)}")


def load_hsi_patches(mat_path, patch_size):
    """
    Load Houston HSI and extract centered patches of size patch_size × patch_size
    around each labeled pixel. Returns arrays:
      X_tr: (N_tr, B, PS, PS), y_tr: (N_tr,),
      X_te: (N_te, B, PS, PS), y_te: (N_te,)
    """
    data = scipy.io.loadmat(mat_path)
    hsi        = data['hsi']      
    train_mask = data['train']   
    test_mask  = data['test']  
    H, W, B = hsi.shape

    pad = patch_size // 2
    hsi_pad = np.pad(hsi, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    tr_pos = np.argwhere(train_mask > 0)
    te_pos = np.argwhere(test_mask  > 0)

    N_tr = tr_pos.shape[0]
    N_te = te_pos.shape[0]
    X_tr = np.zeros((N_tr, B, patch_size, patch_size), dtype=hsi.dtype)
    y_tr = np.zeros(N_tr, dtype=int)
    X_te = np.zeros((N_te, B, patch_size, patch_size), dtype=hsi.dtype)
    y_te = np.zeros(N_te, dtype=int)

  
    for i, (r, c) in enumerate(tr_pos):
        patch = hsi_pad[r:r+patch_size, c:c+patch_size, :]
        X_tr[i] = patch.transpose(2, 0, 1)  # (B, PS, PS)
        y_tr[i] = train_mask[r, c] - 1

    for i, (r, c) in enumerate(te_pos):
        patch = hsi_pad[r:r+patch_size, c:c+patch_size, :]
        X_te[i] = patch.transpose(2, 0, 1)
        y_te[i] = test_mask[r, c] - 1

    return X_tr, y_tr, X_te, y_te


def init_encoder(cfg, device):
    """
    Initialize IJ-EPA encoder, override patch_embed for 144-band input,
    and return the encoder only (frozen later by eval()).
    """
    meta = cfg['meta']
    mask = cfg['mask']
    data = cfg['data']

    encoder, _ = init_model(
        device=device,
        patch_size=mask['patch_size'],
        crop_size=data['crop_size'],
        pred_depth=meta['pred_depth'],
        pred_emb_dim=meta['pred_emb_dim'],
        model_name=meta['model_name']
    )

    proj = encoder.patch_embed.proj
    print(f"DEBUG: original patch_embed.proj = {proj}")
    encoder.patch_embed.proj = torch.nn.Conv2d(
        in_channels=144,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=(proj.bias is not None)
    )
    torch.nn.init.kaiming_uniform_(encoder.patch_embed.proj.weight, nonlinearity='relu')
    if proj.bias is not None:
        torch.nn.init.constant_(encoder.patch_embed.proj.bias, 0)
    print(f"DEBUG: overridden patch_embed.proj = {encoder.patch_embed.proj}")

    return encoder

@torch.no_grad()
def extract_features(encoder, X, device, batch_size=512):
    """
    Extract features by applying the convolutional patch_embed.proj layer,
    then running tokens through transformer blocks and layer norm,
    finally global-average-pooling.
    Debug prints conv output shape for each batch.
    """
    encoder.eval()
    feats = []
    conv = encoder.patch_embed.proj
    for i in range(0, X.shape[0], batch_size):
        batch_np = X[i:i+batch_size]
        batch = torch.from_numpy(batch_np).float().to(device)
        # Conv2d projection: (B, 144, PS, PS) -> (B, C, H', W')
        x = conv(batch)
        print(f"DEBUG: conv output shape = {x.shape}")
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in encoder.blocks:
            x = blk(x)
        x = encoder.norm(x)
        feat = x.mean(dim=1)
        feats.append(feat.cpu())
    feats = torch.cat(feats, dim=0).numpy()
    return feats


def train_and_eval(feats_tr, y_tr, feats_te, y_te):
    clf = LogisticRegression(
        multi_class='multinomial', max_iter=1000,
        C=1.0, class_weight='balanced'
    )
    clf.fit(feats_tr, y_tr)
    y_pred = clf.predict(feats_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, digits=4)
    print(f"Overall accuracy: {acc:.4f}\n")
    print("Classification report:\n", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to IJ-EPA YAML config")
    parser.add_argument('--mat', type=str, required=True,
                        help="Path to houston_data.mat")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="Path to IJ-EPA checkpoint (.pth.tar)")
    parser.add_argument('--batch_size', type=int, default=512,
                        help="Batch size for feature extraction")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    patch_size = cfg['mask']['patch_size']
    print(f"DEBUG: using patch_size = {patch_size}")

    print("Loading HSI patches...")
    X_tr, y_tr, X_te, y_te = load_hsi_patches(args.mat, patch_size)
    print(f"DEBUG: X_tr.shape = {X_tr.shape}, X_te.shape = {X_te.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = init_encoder(cfg, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(ckpt['encoder'], strict=True)
    encoder.to(device)

    bs = args.batch_size
    print(f"Extracting {X_tr.shape[0]} train features in batches of {bs}...")
    feats_tr = extract_features(encoder, X_tr, device, bs)
    print(f"Extracting {X_te.shape[0]} test features in batches of {bs}...")
    feats_te = extract_features(encoder, X_te, device, bs)

    train_and_eval(feats_tr, y_tr, feats_te, y_te)


if __name__ == '__main__':
    main()
