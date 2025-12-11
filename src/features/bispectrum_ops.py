import numpy as np
from scipy.stats import skew, kurtosis


def get_bispectrum_features(image: np.ndarray, block_size=32) -> list:
    """
    Approximates Bispectral features using statistical moments of patches.
    Returns: [skewness, kurtosis, variance_of_patches]
    """
    h, w = image.shape
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size

    patches = []
    for i in range(0, h_trim, block_size):
        for j in range(0, w_trim, block_size):
            patch = image[i:i + block_size, j:j + block_size]
            patches.append(patch.flatten())

    if not patches:
        return [0.0, 0.0, 0.0]

    flat_data = np.array(patches).flatten()
    val_skew = skew(flat_data)
    val_kurt = kurtosis(flat_data)
    patch_means = [np.mean(p) for p in patches]
    val_spatial_var = np.var(patch_means)

    return [val_skew, val_kurt, val_spatial_var]