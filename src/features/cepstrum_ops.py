import numpy as np
from scipy import fftpack


def get_cepstrum_features(image: np.ndarray) -> list:
    """
    Computes Cepstral features to detect periodic artifacts.
    Returns: [mean_high_quefrency, kurtosis_high_quefrency]
    """
    f = fftpack.fft2(image)
    log_spectrum = np.log(np.abs(f) + 1e-8)
    cepstrum = np.abs(fftpack.ifft2(log_spectrum))
    cepstrum = fftpack.fftshift(cepstrum)

    h, w = cepstrum.shape
    cy, cx = h // 2, w // 2
    r = 10  # Filter radius

    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    mask = x * x + y * y > r * r

    high_quefrency = cepstrum[mask]

    cep_kurtosis = 0.0
    if len(high_quefrency) > 0:
        mean_val = np.mean(high_quefrency)
        std_val = np.std(high_quefrency)
        if std_val > 0:
            cep_kurtosis = np.mean(((high_quefrency - mean_val) / std_val) ** 4)

    cep_mean = np.mean(high_quefrency)

    return [cep_mean, cep_kurtosis]