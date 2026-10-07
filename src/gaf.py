import numpy as np
from scipy.signal import resample


def resample_signal(signal, size):
    """Resample a 1D signal to ``size`` temporal points."""

    return resample(signal, size)


def normalize_to_minus_one_one(signal):
    """Convert a signal from [0, 1] to [-1, 1]."""

    return 2.0 * signal - 1.0


def create_gasf(signal):
    """Generate a Gramian Angular Summation Field."""

    values = np.clip(signal, -1.0, 1.0)
    theta = np.arccos(values)

    return np.cos(theta[:, None] + theta[None, :])


def create_gadf(signal):
    """Generate a Gramian Angular Difference Field."""

    values = np.clip(signal, -1.0, 1.0)
    theta = np.arccos(values)

    return np.sin(theta[:, None] - theta[None, :])


def signal_to_gaf(signal, image_size=224):
    """Convert a normalized [0, 1] signal to GASF and GADF arrays."""

    reduced = resample_signal(signal, image_size)
    reduced = np.clip(reduced, 0.0, 1.0)
    reduced = normalize_to_minus_one_one(reduced)

    gasf = create_gasf(reduced)
    gadf = create_gadf(reduced)

    return gasf.astype(np.float32), gadf.astype(np.float32)
