import tifffile
import matplotlib.pyplot as plt
import numpy as np

file_path = "/Users/zainab/HSI using jepa/dataset/19920612_AVIRIS_IndianPine_Site3.tif"
image = tifffile.imread(file_path)
print(f"Image shape: {image.shape}")  # (220, 145, 145)

band_number = 219
plt.imshow(image[band_number], cmap='gray')
plt.title(f"Band {band_number}")
plt.colorbar()
plt.show()

rgb = np.stack([image[50], image[30], image[20]], axis=-1)
rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))  # Normalize
plt.imshow(rgb)
plt.title("False-color RGB (Bands 50, 30, 20)")
plt.show()
