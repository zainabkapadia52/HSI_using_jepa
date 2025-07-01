from src.datasets.hsitile import HsiTifTileDataset
from src.transforms      import make_hyperspec_transforms

ds = HsiTifTileDataset(
    tif_path   = "/Users/zainab/HSI_using_jepa/dataset/19920612_AVIRIS_IndianPine_Site3.tif",
    patch_size = 64, stride=32, bands=144,
    transform  = make_hyperspec_transforms(64)
)

# your assert
assert ds[0].shape == (144,64,64)

# a friendly print
print("✅ Tile 0 shape:", ds[0].shape)
print("Total tiles:", len(ds))
