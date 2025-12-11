import numpy as np
from scipy import fftpack
from scipy.stats import linregress


def get_raps(image: np.ndarray) -> np.ndarray:
    """
    Computes Radially Averaged Power Spectrum (RAPS).

    This compresses the 2D frequency information into a 1D signature.
    Real cameras have a specific '1/f' decay profile.
    Generative models (Flux/SDXL) often 'drop off' faster at high frequencies.
    """
    # 1. Compute 2D FFT
    # Convert image to float to prevent overflow
    image = image.astype(float)
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)

    # Compute Power Spectrum (Magnitude Squared)
    magnitude = np.abs(fshift) ** 2

    # 2. Setup Radial Coordinate System
    center = np.array(image.shape) // 2
    y, x = np.indices(image.shape)

    # Calculate distance (radius) from center for every pixel
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

    # 3. Radial Averaging (The "Compression" Step)
    # Sum up energy in each ring
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    # Count pixels in each ring
    nr = np.bincount(r.ravel())

    # Average energy per ring
    # (Avoid division by zero with maximum(nr, 1))
    radial_profile = tbin / np.maximum(nr, 1)

    # 4. Normalize (0 to 1) for scale invariance
    if np.max(radial_profile) > 0:
        radial_profile = radial_profile / np.max(radial_profile)

    return radial_profile


def get_spectral_slope(raps: np.ndarray) -> float:
    """
    Calculates the slope (alpha) of the log-log power spectrum.

    Physics Principle:
    Natural images follow a '1/f^alpha' power law, meaning they form
    a straight line when plotted on a log-log graph.

    Deepfakes often violate this law (steeper slope or non-linear curves)
    due to the 'smoothing' effect of the ODE solver.
    """
    # 1. Pre-processing
    # Clip to avoid log(0) errors if energy is perfectly zero
    raps = np.clip(raps, 1e-10, None)

    # 2. Setup Axes
    # We skip index 0 (DC component/Average brightness) as it distorts the slope
    freqs = np.arange(1, len(raps))
    power = raps[1:]

    # 3. Log-Log Transform
    log_freqs = np.log10(freqs)
    log_power = np.log10(power)

    # 4. Linear Regression
    # Fits a straight line to the data: y = mx + c
    slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_power)

    # The slope is usually negative (energy goes down as freq goes up).
    # We return it directly.
    return slope