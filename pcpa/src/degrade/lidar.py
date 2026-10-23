"""LiDAR degradation: attenuation, spurious returns, beam dilation.

Boreas stores each LiDAR point as six float32 values ``(x, y, z, intensity,
range, time)``, which is the layout this module expects; only the first
three columns carry the geometry.

Three effects are modelled, the ones the adverse-weather literature
consistently reports:

* attenuation of the returned power, ``P ~ exp(-2 beta d)``, which is what
  makes wet and foggy returns fade with distance;
* spurious returns from back-scattered particles, which show up as points at
  the wrong range;
* beam dilation, a growing angular footprint that smears the geometry.

This is a parametric stand-in for the physically based fog simulation of
Hahner et al. (ICCV 2021). The parametric form is used when a controlled
severity axis has to be swept; the physical model is the reference the paper
cites for the degradation itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarDegradation:
    """Parameters of one LiDAR degradation level."""

    extinction_beta_per_m: float = 0.0
    spurious_return_rate: float = 0.0     # spurious points per real point
    beam_dilation_rad: float = 0.0
    range_noise_frac: float = 0.0         # noise std as a fraction of range
    max_range_m: float = 200.0
    spurious_intensity: float = 30.0


def ranges(points: np.ndarray) -> np.ndarray:
    """Slant range of every point."""
    return np.linalg.norm(np.asarray(points, dtype=float)[:, :3], axis=1)


def attenuate(
    points: np.ndarray,
    extinction_beta_per_m: float,
    rng: np.random.Generator,
    range_noise_frac: float = 0.0,
) -> np.ndarray:
    """Fade the returned power with distance and jitter the range."""
    if extinction_beta_per_m <= 0.0 and range_noise_frac <= 0.0:
        return np.asarray(points, dtype=float).copy()
    out = np.asarray(points, dtype=float).copy()
    distance = ranges(out)
    if extinction_beta_per_m > 0.0 and out.shape[1] > 3:
        out[:, 3] *= np.exp(-2.0 * extinction_beta_per_m * distance)
    if range_noise_frac > 0.0:
        out[:, :3] += rng.normal(0.0, 1.0, size=out[:, :3].shape) * (
            range_noise_frac * distance)[:, None]
    return out


def spurious_returns(
    points: np.ndarray,
    rate: float,
    rng: np.random.Generator,
    max_range_m: float = 200.0,
    intensity: float = 30.0,
) -> np.ndarray:
    """Append returns from particles inside the beam.

    The directions are borrowed from real points (the beam really did point
    there) while the ranges are drawn uniformly, so the additions look like
    back-scatter rather than like a shifted copy of the scene.
    """
    points = np.asarray(points, dtype=float)
    count = int(round(rate * points.shape[0]))
    if count <= 0 or points.shape[0] == 0:
        return points.copy()
    source = points[rng.integers(0, points.shape[0], size=count), :3]
    direction = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-9)
    distance = rng.uniform(2.0, max_range_m, size=count)
    extra = np.zeros((count, points.shape[1]), dtype=float)
    extra[:, :3] = direction * distance[:, None]
    if points.shape[1] > 3:
        extra[:, 3] = intensity
    if points.shape[1] > 4:
        extra[:, 4] = distance
    return np.vstack([points, extra])


def beam_dilation(
    points: np.ndarray,
    sigma_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Smear each point perpendicular to its ray, growing with range."""
    if sigma_rad <= 0.0:
        return np.asarray(points, dtype=float).copy()
    out = np.asarray(points, dtype=float).copy()
    xyz = out[:, :3]
    distance = np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-9)
    direction = xyz / distance
    jitter = rng.normal(0.0, 1.0, size=xyz.shape)
    jitter -= np.sum(jitter * direction, axis=1, keepdims=True) * direction
    out[:, :3] = xyz + jitter * (sigma_rad * distance)
    return out


def degrade(
    points: np.ndarray,
    params: LidarDegradation,
    rng: np.random.Generator,
) -> np.ndarray:
    """Full LiDAR degradation chain at one severity level."""
    out = attenuate(points, params.extinction_beta_per_m, rng, params.range_noise_frac)
    out = spurious_returns(out, params.spurious_return_rate, rng,
                           params.max_range_m, params.spurious_intensity)
    out = beam_dilation(out, params.beam_dilation_rad, rng)
    return out


def summary(points: np.ndarray) -> dict[str, float]:
    """Descriptive statistics used to check that degradation did something."""
    points = np.asarray(points, dtype=float)
    distance = ranges(points)
    stats = {
        "n_points": float(points.shape[0]),
        "max_range_m": float(distance.max()) if distance.size else 0.0,
    }
    if points.shape[1] > 3:
        stats["mean_intensity"] = float(np.mean(points[:, 3]))
    return stats
