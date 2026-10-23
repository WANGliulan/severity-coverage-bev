"""Radar degradation: clutter, angular smearing, speckle.

Boreas distributes the Navtech radar as polar images, one frame per
range-azimuth sweep, so a frame is an array whose rows are azimuth bins and
whose columns are range bins. Everything here therefore works on a 2-D
array and treats axis 0 as azimuth.

Three effects are modelled:

* clutter: back-scatter from precipitation raises the noise floor;
* angular smearing: the azimuthal point-spread function widens, which is the
  resolution loss the fusion literature reports for radar;
* speckle: the coherent field is multiplied by a Rayleigh amplitude, or
  averaged over ``looks`` independent samples to reduce it.

Because azimuth is periodic, the angular blur wraps around rather than
bleeding off the edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class RadarDegradation:
    """Parameters of one radar degradation level."""

    clutter_to_signal_db: float = 0.0
    angular_smear_bins: float = 0.0
    looks: float = 1.0            # 1 = fully developed speckle
    dynamic_range_db: float = 40.0


def to_float(frame: np.ndarray) -> np.ndarray:
    """Accept uint8 or float input and return float in [0, 1]."""
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.size and arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def add_clutter(
    frame: np.ndarray,
    clutter_to_signal_db: float,
    rng: np.random.Generator,
    dynamic_range_db: float = 40.0,
) -> np.ndarray:
    """Raise the noise floor by a clutter-to-signal ratio.

    The clutter amplitude is ``10 ** (-csr/20)`` of the full dynamic range,
    which is the usual way a C/S ratio in dB is turned into a scale factor.
    """
    frame = to_float(frame)
    if clutter_to_signal_db <= 0.0:
        return frame.copy()
    # The full dynamic range maps onto the unit interval, so a ratio in dB
    # is 10**(csr/20) times the smallest representable level. A larger
    # clutter-to-signal ratio must therefore mean MORE clutter.
    smallest_level = 10.0 ** (-dynamic_range_db / 20.0)
    level = smallest_level * 10.0 ** (clutter_to_signal_db / 20.0)
    clutter = rng.rayleigh(scale=level, size=frame.shape).astype(np.float32)
    return np.clip(frame + clutter, 0.0, 1.0)


def angular_smear(frame: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Blur along the (periodic) azimuth axis."""
    frame = to_float(frame)
    if sigma_bins <= 0.0:
        return frame.copy()
    return gaussian_filter1d(frame, sigma=sigma_bins, axis=0, mode="wrap")


def speckle(
    frame: np.ndarray,
    rng: np.random.Generator,
    looks: float = 1.0,
) -> np.ndarray:
    """Multiply by a Gamma-distributed coherent amplitude.

    ``looks = 1`` gives fully developed speckle; larger values model the
    averaging that a longer integration performs.
    """
    frame = to_float(frame)
    if looks <= 0.0:
        raise ValueError("looks must be positive")
    amplitude = rng.gamma(shape=looks, scale=1.0 / looks, size=frame.shape)
    return np.clip(frame * amplitude.astype(np.float32), 0.0, 1.0)


def degrade(
    frame: np.ndarray,
    params: RadarDegradation,
    rng: np.random.Generator,
) -> np.ndarray:
    """Full radar degradation chain at one severity level."""
    out = to_float(frame)
    if params.angular_smear_bins > 0.0:
        out = angular_smear(out, params.angular_smear_bins)
    out = speckle(out, rng, params.looks)
    out = add_clutter(out, params.clutter_to_signal_db, rng, params.dynamic_range_db)
    return out


def summary(frame: np.ndarray) -> dict[str, float]:
    """Descriptive statistics used to check that degradation did something."""
    frame = to_float(frame)
    if frame.size == 0:
        return {"mean": 0.0, "std": 0.0, "floor": 0.0, "ceiling": 0.0,
                "dynamic_range": 0.0}
    low, high = np.percentile(frame, [5.0, 99.0])
    return {
        "mean": float(np.mean(frame)),
        "std": float(np.std(frame)),
        "floor": float(low),            # rises with clutter
        "ceiling": float(high),
        "dynamic_range": float(high - low),
    }
