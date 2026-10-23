"""Road-surface labels for the Boreas sequences.

Boreas ships no per-frame weather annotation, so the surface state has to be
established by hand. This module does two things.

* It reads and writes a hand-maintained mapping in
  ``config/boreas_weather.yaml``: one label per *sequence*, drawn from the
  vocabulary of ``surface.friction_mu_by_weather`` in the main configuration.
* It computes a **data-driven proxy** per sequence, so that the labelling is
  not made blind. Rain and snow leave measurable traces in a LiDAR sweep ---
  attenuated long-range returns, a growing share of weak returns, spurious
  points from particles in the beam --- and those statistics are worth looking
  at before spending hours on video.

The proxy is explicitly *not* a label. It ranks the sequences so that the ones
worth inspecting first become obvious, and it is reported next to the label so
that a reader can see whether the two agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .boreas import BoreasSequence, load_lidar_frame

#: The vocabulary a sequence label may take. Kept identical to the keys of
#: ``surface.friction_mu_by_weather`` so that a label implies a reference
#: friction coefficient without any further mapping.
WEATHER_VOCABULARY = (
    "dry", "wet", "light_rain", "heavy_rain", "snow", "packed_snow", "ice",
)

FAR_RANGE_M = 40.0
LOW_INTENSITY = 20.0


def default_weather_map_path() -> Path:
    from ..common.config import project_root
    return project_root() / "config" / "boreas_weather.yaml"


def load_weather_map(path: str | Path | None = None) -> dict[str, str]:
    """Read the sequence -> weather mapping; missing file means empty."""
    path = Path(path) if path else default_weather_map_path()
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    mapping = data.get("sequences", data) or {}
    return {str(k): str(v) for k, v in mapping.items()}


#: The vocabulary a *weather condition* label may take. Distinct from
#: :data:`WEATHER_VOCABULARY`, which describes the road surface: a sequence can be
#: snow-covered without snow falling, and the two questions need different labels.
CONDITION_VOCABULARY = ("clear", "rain", "snowfall", "night", "low_sun", "fog")


def default_conditions_path() -> Path:
    from ..common.config import project_root
    return project_root() / "config" / "boreas_conditions.yaml"


def load_conditions(path: str | Path | None = None) -> dict[str, str]:
    """Read the sequence -> weather-condition mapping; missing file means empty.

    Conditions describe the weather in the scene rather than the state of the
    driving surface, so they live in their own file: the surface labels imply a
    friction reference and must not be overloaded with sky conditions.
    """
    path = Path(path) if path else default_conditions_path()
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    mapping = data.get("sequences", data) or {}
    return {str(k): str(v) for k, v in mapping.items()}


def save_weather_map(mapping: dict[str, str], path: str | Path | None = None) -> Path:
    """Write the mapping back, preserving the explanatory header."""
    path = Path(path) if path else default_weather_map_path()
    header = (
        "# Sequence -> road-surface label, filled in by hand.\n"
        "#\n"
        "# Boreas has no weather annotation, so this file is the label source for\n"
        "# the cross-weather split and for the reference friction coefficient.\n"
        "# Allowed labels: " + ", ".join(WEATHER_VOCABULARY) + "\n"
        "#\n"
        "# Use scripts/label_boreas_weather.py to see the data-driven proxy\n"
        "# for each sequence, then look at real camera frames with scripts/make_weather_sheets.py.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump({"sequences": mapping}, handle, allow_unicode=True, sort_keys=True)
    return path


def sequence_proxy(sequence: BoreasSequence, max_frames: int = 12,
                   stride: int = 40) -> dict[str, float]:
    """LiDAR statistics that track adverse weather, for one sequence.

    Sampled rather than exhaustive: a dozen frames spread through the sequence
    are enough to tell a snowy sweep from a clear one, and it keeps the tool
    usable while a download is still running.
    """
    paths = sequence.frames("lidar")
    if not paths:
        return {}
    taken = paths[::stride][:max_frames] if stride > 1 else paths[:max_frames]

    far_fraction, far_intensity, low_fraction = [], [], []
    ranges_p99, counts, intensities = [], [], []
    for path in taken:
        points = load_lidar_frame(path)
        if points.size == 0:
            continue
        distance = np.linalg.norm(points[:, :3], axis=1)
        intensity = points[:, 3] if points.shape[1] > 3 else np.zeros_like(distance)
        ranges_p99.append(float(np.percentile(distance, 99.0)))
        counts.append(float(points.shape[0]))
        intensities.append(float(intensity.mean()))
        far = distance > FAR_RANGE_M
        far_fraction.append(float(far.mean()))
        far_intensity.append(float(intensity[far].mean()) if far.any() else 0.0)
        low_fraction.append(float((intensity < LOW_INTENSITY).mean()))

    if not counts:
        return {}
    return {
        "frames_sampled": float(len(counts)),
        "points_mean": float(np.mean(counts)),
        "range_p99_mean_m": float(np.mean(ranges_p99)),
        "intensity_mean": float(np.mean(intensities)),
        "far_fraction": float(np.mean(far_fraction)),
        "far_intensity": float(np.mean(far_intensity)),
        "low_intensity_fraction": float(np.mean(low_fraction)),
    }


def proxy_scores(rows: list[dict[str, float]]) -> np.ndarray:
    """A single 'how adverse does this look' score per row, higher = worse.

    Combines the two signatures that adverse weather leaves behind: fewer
    returns survive out to far range, and a larger share of the returns are
    weak. Standardised so the two contribute comparably.
    """
    if not rows:
        return np.zeros(0)
    far = np.array([row.get("far_fraction", np.nan) for row in rows], dtype=float)
    low = np.array([row.get("low_intensity_fraction", np.nan) for row in rows], dtype=float)

    def standardise(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if finite.sum() < 2:
            return np.zeros_like(values)
        mean, std = values[finite].mean(), values[finite].std()
        if std < 1e-9:
            return np.zeros_like(values)
        return (values - mean) / std

    return standardise(low) - standardise(far)
