# src/datasets/houston13.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from logging import getLogger

logger = getLogger()
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)

def make_houston13(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=0,       # keep at 0 to avoid multiprocessing/pickle issues
    world_size=1,
    rank=0,
    root_path=None,
    image_folder='patches',
    drop_last=True
):
    """
    Creates a Houston13Dataset and DataLoader for distributed IJepa training.

    Args:
      transform: a torch transform expecting a tensor of shape [C, H, W].
      batch_size: number of samples per batch.
      collator: MaskCollator instance to build masked batches.
      pin_mem: DataLoader pin_memory flag.
      num_workers: number of worker processes (0 avoids pickle issues).
      world_size, rank: for distributed sampling.
      root_path: base directory containing `image_folder` subdir.
      image_folder: subdirectory under root_path with .npy files.
      drop_last: whether to drop the last incomplete batch.
    """
    dataset = Houston13Dataset(root_path, image_folder, transform)
    sampler = distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info(f'Houston13 dataset loaded: {len(dataset)} samples')
    return dataset, loader, sampler


class Houston13Dataset(Dataset):
    """
    PyTorch Dataset for Houston 2013 hyperspectral patches stored as .npy files.
    Each file is a numpy array of shape either [H, W, bands] or [bands, H, W].
    Returns a FloatTensor of shape [bands=144, H, W].
    """
    def __init__(self, root, image_folder='patches', transform=None):
        self.data_dir = os.path.join(root, image_folder)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Houston13 data dir not found: {self.data_dir}")
        self.files = sorted([f for f in os.listdir(self.data_dir) if f.lower().endswith('.npy')])
        self.transform = transform
        logger.info(f'Found {len(self.files)} .npy patches in {self.data_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.data_dir, fname)
        img = np.load(path).astype(np.float32)
        if img.ndim == 3 and img.shape[0] == 144:
            hs = img[:144, :, :]           # [144, H, W]
        elif img.ndim == 3 and img.shape[2] == 144:
            # It was [H, W, bands=144], so transpose:
            hs = img[:, :, :144]          # [H, W, 144]
            hs = np.transpose(hs, (2, 0, 1))  # → [144, H, W]
        else:
            raise ValueError(f"Unexpected patch shape {img.shape}; expected (144,H,W) or (H,W,144).")
        
        tensor = torch.from_numpy(hs).float()
        if self.transform is not None:
            tensor = self.transform(tensor)

        return (tensor,)

# EOF
