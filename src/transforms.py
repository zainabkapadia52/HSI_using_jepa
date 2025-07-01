import random
import torch
from logging import getLogger
import torchvision.transforms.functional as TF

logger = getLogger()

class RandomResizedCropTensor:
    """
    Randomly crop a tensor [C,H,W] and resize to given size.
    """
    def __init__(self, size, scale=(0.8, 1.0), ratio=(1.0, 1.0), attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.attempts = attempts

    def __call__(self, tensor):
        # tensor: [C, H, W]
        C, H, W = tensor.shape
        area = H * W
        for _ in range(self.attempts):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect = random.uniform(self.ratio[0], self.ratio[1])
            h = int(round((target_area / aspect) ** 0.5))
            w = int(round((target_area * aspect) ** 0.5))
            if h <= H and w <= W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                patch = tensor[:, top:top+h, left:left+w]
                # resize with interpolation
                patch = TF.resize(patch.unsqueeze(0), (self.size, self.size)).squeeze(0)
                return patch
        # fallback center crop + resize
        center_top = (H - self.size) // 2
        center_left = (W - self.size) // 2
        patch = tensor[:, center_top:center_top+self.size, center_left:center_left+self.size]
        return patch

class RandomHorizontalFlipTensor:
    """
    Horizontally flip a tensor [C,H,W] with probability p.
    """
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, tensor):
        if random.random() < self.p:
            return tensor.flip(-1)
        return tensor

class NormalizeTensor:
    """
    Normalize tensor [C,H,W] with per-channel mean/std lists.
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

class BandDropout:
    """
    Randomly zero-out `n` spectral bands in the tensor.
    """
    def __init__(self, n_drop):
        self.n_drop = n_drop
    def __call__(self, tensor):
        C, H, W = tensor.shape
        drop_idx = random.sample(range(C), k=min(self.n_drop, C))
        tensor[drop_idx, :, :] = 0
        return tensor

class SpectralNoise:
    """
    Add Gaussian noise (σ) per spectral band.
    """
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise


def make_transforms(
    crop_size=19,
    crop_scale=(0.8,1.0),
    horizontal_flip=False,
    normalize_mean=None,
    normalize_std=None,
    band_dropout=0,
    spectral_noise=0.0
):
    """
    Compose transforms for hyperspectral patches.
    """
    transforms = []
    transforms.append(RandomResizedCropTensor(crop_size, scale=crop_scale))
    if horizontal_flip:
        transforms.append(RandomHorizontalFlipTensor(p=0.5))
    if band_dropout > 0:
        transforms.append(BandDropout(band_dropout))
    if spectral_noise > 0.0:
        transforms.append(SpectralNoise(spectral_noise))
    if normalize_mean is not None and normalize_std is not None:
        transforms.append(NormalizeTensor(normalize_mean, normalize_std))

    def _transform(x):
        for t in transforms:
            x = t(x)
        return x
    return _transform
