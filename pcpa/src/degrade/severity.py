"""The five-level severity ladder.

The protocol defines levels ``s = 0 .. 4``, where ``s = 0`` is the
undegraded recording. Each modality parameter in ``config/default.yaml``
holds one value per level, and :meth:`SeverityLadder.at` interpolates the
ladder at any (possibly fractional) level, so a figure swept over ``s`` is
reproducible from the configuration alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModalityParameters:
    """The per-modality knobs of one severity level."""

    camera_beta_per_m: float
    camera_streak_density: float
    camera_exposure_loss: float
    camera_read_noise_std: float
    lidar_beta_per_m: float
    lidar_spurious_rate: float
    lidar_beam_dilation_rad: float
    lidar_range_noise_frac: float
    radar_clutter_db: float
    radar_angular_smear_bins: float
    radar_looks: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class SeverityLadder:
    """Per-level parameter arrays, plus interpolation."""

    camera_beta_per_m: np.ndarray
    camera_streak_density: np.ndarray
    camera_exposure_loss: np.ndarray
    camera_read_noise_std: np.ndarray
    lidar_beta_per_m: np.ndarray
    lidar_spurious_rate: np.ndarray
    lidar_beam_dilation_rad: np.ndarray
    lidar_range_noise_frac: np.ndarray
    radar_clutter_db: np.ndarray
    radar_angular_smear_bins: np.ndarray
    radar_looks: np.ndarray

    @classmethod
    def from_config(cls, cfg) -> "SeverityLadder":
        camera = cfg["degradation.camera"]
        lidar = cfg["degradation.lidar"]
        radar = cfg["degradation.radar"]
        ladder = cls(
            camera_beta_per_m=np.asarray(camera["extinction_beta_per_m"], dtype=float),
            camera_streak_density=np.asarray(camera["streak_density"], dtype=float),
            camera_exposure_loss=np.asarray(camera["exposure_loss"], dtype=float),
            camera_read_noise_std=np.asarray(camera["read_noise_std"], dtype=float),
            lidar_beta_per_m=np.asarray(lidar["extinction_beta_per_m"], dtype=float),
            lidar_spurious_rate=np.asarray(lidar["spurious_return_rate"], dtype=float),
            lidar_beam_dilation_rad=np.asarray(lidar["beam_dilation_rad"], dtype=float),
            lidar_range_noise_frac=np.asarray(lidar["range_noise_frac"], dtype=float),
            radar_clutter_db=np.asarray(radar["clutter_to_signal_db"], dtype=float),
            radar_angular_smear_bins=np.asarray(radar["angular_smear_bins"], dtype=float),
            radar_looks=np.asarray(radar["looks"], dtype=float),
        )
        lengths = {name: arr.size for name, arr in ladder.arrays().items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"severity levels are inconsistent across modalities: {lengths}")
        return ladder

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def levels(self) -> np.ndarray:
        return np.arange(self.camera_beta_per_m.size, dtype=float)

    def at(self, level: float) -> ModalityParameters:
        """Parameters at ``level``, linearly interpolated between samples."""
        grid = self.levels
        level = float(np.clip(level, grid[0], grid[-1]))
        values = {name: float(np.interp(level, grid, arr)) for name, arr in self.arrays().items()}
        return ModalityParameters(**values)

    def table(self) -> str:
        params = [self.at(float(s)) for s in self.levels]
        head = (f"{'s':>2}{'cam_beta':>10}{'streaks':>9}{'expose':>8}{'noise':>7}"
                f"{'lidar_beta':>12}{'spurious':>10}{'radar_dB':>10}")
        lines = [head, "-" * len(head)]
        for s, p in zip(self.levels, params):
            lines.append(f"{int(s):>2}{p.camera_beta_per_m:>10.4f}"
                         f"{p.camera_streak_density:>9.2f}"
                         f"{p.camera_exposure_loss:>8.2f}"
                         f"{p.camera_read_noise_std:>7.3f}"
                         f"{p.lidar_beta_per_m:>12.4f}"
                         f"{p.lidar_spurious_rate:>10.3f}{p.radar_clutter_db:>10.1f}")
        return "\n".join(lines)
