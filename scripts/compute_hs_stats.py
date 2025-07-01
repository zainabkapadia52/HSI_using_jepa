import os
import numpy as np

data_dir = "/Users/zainab/HSI_using_jepa/dataset/Houston13_patches"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
sum_bands = None
sum_sq_bands = None
total_pixels = 0

for fname in files:
    path = os.path.join(data_dir, fname)
    arr = np.load(path).astype(np.float64)    # [C, H, W]
    C, H, W = arr.shape
    
    C = min(C, 144)
    arr = arr[:C, :, :]                       # [C, H, W]
    
    arr_flat = arr.reshape(C, -1)             # [C, H*W]
    
    if sum_bands is None:
        sum_bands    = arr_flat.sum(axis=1)   # shape [C]
        sum_sq_bands = (arr_flat**2).sum(axis=1)
    else:
        sum_bands    += arr_flat.sum(axis=1)
        sum_sq_bands += (arr_flat**2).sum(axis=1)
    
    total_pixels += H * W                    # number of samples per channel
    
mean = sum_bands / total_pixels             # shape [C]
var  = (sum_sq_bands / total_pixels) - (mean**2)
std  = np.sqrt(var)

print(f"Found {len(mean)} bands.")
print("Per-band mean:")
print(mean.tolist())
print("\nPer-band std:")
print(std.tolist())
 