"""Camera degradation: atmospheric scattering, rain streaks, low light.

Koschmieder's law relates the scene radiance ``J`` to the observed radiance
``I`` through a scattering medium::

    I = J * t + A * (1 - t),     t = exp(-beta * d)

with ``t`` the transmittance along the line of sight, ``d`` the distance to
the surface and ``A`` the atmospheric light. The extinction coefficient
``beta`` is what the severity ladder drives.

Without a depth map the distance is a scalar (``scene_depth_m``) --- a crude
but explicit assumption that makes the degradation uniform over the frame
rather than depth dependent. Pass a per-pixel distance array to get the
realistic, depth-dependent version.

Low light is modelled by two terms rather than one. ``exposure_loss`` scales the
radiance down, and ``read_noise_std`` adds noise at a fixed magnitude, so that
the signal-to-noise ratio falls as the illumination does --- which is what a real
sensor does. Scaling alone would be a change any contrast-normalising model could
undo, and would leave the perception problem untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraDegradation:
    """Parameters of one camera degradation level."""

    extinction_beta_per_m: float = 0.0
    streak_density: float = 0.0
    exposure_loss: float = 0.0        # 0 = none, 1 = fully dark
    read_noise_std: float = 0.0       # sensor noise, in units of full scale
    scene_depth_m: float = 30.0
    atmospheric_light: float = 0.85   # grey-white haze
    streak_length_px: int = 24
    streak_tilt_deg: float = 12.0


def to_float_image(image: np.ndarray) -> np.ndarray:
    """Accept uint8 or float input and return float in [0, 1]."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.size and arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def koschmieder(
    image: np.ndarray,
    extinction_beta_per_m: float,
    distance_m: np.ndarray | float,
    atmospheric_light: float = 0.85,
) -> np.ndarray:
    """Apply atmospheric scattering to a float image in [0, 1]."""
    image = to_float_image(image)
    if extinction_beta_per_m < 0.0:
        raise ValueError("extinction coefficient must be non-negative")
    transmittance = np.exp(-extinction_beta_per_m * np.asarray(distance_m, dtype=np.float32))
    while transmittance.ndim < image.ndim:
        transmittance = transmittance[..., None]
    return np.clip(image * transmittance + atmospheric_light * (1.0 - transmittance), 0.0, 1.0)


def rain_streaks(
    image: np.ndarray,
    density: float,
    rng: np.random.Generator,
    length_px: int = 24,
    tilt_deg: float = 12.0,
    brightness: float = 0.55,
) -> np.ndarray:
    """Overlay ``density``-proportional bright streaks (a crude rain model).

    ``density`` is the expected number of streaks per thousand pixels, so the
    same value gives a visually comparable effect across resolutions.
    """
    image = to_float_image(image).copy()
    if density <= 0.0:
        return image
    height, width = image.shape[:2]
    count = int(round(density * height * width / 1000.0))
    if count <= 0:
        return image
    ys = rng.integers(0, height, size=count)
    xs = rng.integers(0, width, size=count)
    tilt = np.deg2rad(tilt_deg)
    dy = int(round(length_px * np.sin(tilt)))
    dx = int(round(length_px * np.cos(tilt)))
    for y, x in zip(ys, xs):
        y_end, x_end = min(y + dy, height), min(x + dx, width)
        if y_end <= y or x_end <= x:
            continue
        segment = image[y:y_end, x:x_end]
        fade = np.linspace(1.0, 0.35, segment.shape[0], dtype=np.float32)[:, None]
        if segment.ndim == 3:
            segment += brightness * fade[..., None]
        else:
            segment += brightness * fade
        np.clip(segment, 0.0, 1.0, out=segment)
    return image


def exposure_loss(image: np.ndarray, amount: float) -> np.ndarray:
    """Scale the radiance down (``amount`` in [0, 1], 1 = black)."""
    image = to_float_image(image)
    return np.clip(image * (1.0 - float(np.clip(amount, 0.0, 1.0))), 0.0, 1.0)


def read_noise(image: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add sensor noise at a fixed magnitude, independent of the illumination.

    This is the term that makes low light hard. A dark scene has a small signal
    and an unchanged noise floor, so the ratio between them collapses; scaling
    the radiance alone would leave a noiseless, merely dim image that a model
    could rescale away.
    """
    out = to_float_image(image)
    if std <= 0.0:
        return out
    noise = rng.normal(0.0, float(std), size=out.shape).astype(np.float32)
    return np.clip(out + noise, 0.0, 1.0)


def degrade(
    image: np.ndarray,
    params: CameraDegradation,
    rng: np.random.Generator,
) -> np.ndarray:
    """Full camera degradation chain at one severity level.

    Order matters: scattering attenuates and veils the scene, rain adds streaks
    on top of the veiled image, and the sensor terms come last --- the exposure
    loss first and then the read noise, because both describe the sensor rather
    than the medium, and because the noise has to be added to the light that
    actually reaches the sensor.
    """
    out = to_float_image(image)
    if params.extinction_beta_per_m > 0.0:
        out = koschmieder(out, params.extinction_beta_per_m, params.scene_depth_m,
                          params.atmospheric_light)
    if params.streak_density > 0.0:
        out = rain_streaks(out, params.streak_density, rng,
                           params.streak_length_px, params.streak_tilt_deg)
    if params.exposure_loss > 0.0:
        out = exposure_loss(out, params.exposure_loss)
    if params.read_noise_std > 0.0:
        out = read_noise(out, params.read_noise_std, rng)
    return out
