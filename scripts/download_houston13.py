from rs_fusion_datasets import fetch_houston2013, split_spmatrix
import numpy as np
import os

hsi, dsm, train_spmat, test_spmat, info = fetch_houston2013()

train_lbl = train_spmat.todense()
test_lbl  = test_spmat.todense()

out_dir = "/Users/zainab/HSI_using_jepa/dataset/Houston13_patches"
os.makedirs(out_dir, exist_ok=True)

patch_sz = 19
pad = patch_sz // 2
H, W = train_lbl.shape

for (lbl_map, split) in [(train_lbl, "train"), (test_lbl, "test")]:
    coords = np.argwhere(np.array(lbl_map) > 0)
    for i, j in coords:
        # Extract a centered patch; skip if you'd go out of bounds
        if i - pad < 0 or i + pad >= H or j - pad < 0 or j + pad >= W:
            continue
        patch = hsi[:, i-pad:i+pad+1, j-pad:j+pad+1]        # [bands,19,19]
        fname = f"{split}_{i}_{j}.npy"
        np.save(os.path.join(out_dir, fname), patch)
