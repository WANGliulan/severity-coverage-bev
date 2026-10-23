"""How much of each recording the downloaded frames actually cover.

Run from the project root::

    .\\.venv\\Scripts\\python.exe scripts\\frame_coverage.py

A download subsamples its sequence, and the question that matters is not how
many frames arrived but *how much of the drive they span*. Sensor frame rates
differ (LiDAR ~10 Hz, camera ~20 Hz, radar ~12 Hz), so taking every n-th key of
each sensor covers three different stretches of time --- and taking the first n
keys of each, as the downloader does, covers the *opening* stretch of all three.

A multi-modal frame is usable only where every sensor has a frame, so the usable
count is set by the shortest of the spans. This script computes all of that from
the applanix ``*_times.txt`` lists, which name every frame of the recording, so
the answer needs no images and no download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config                             # noqa: E402
from src.data.boreas import discover_sequences                        # noqa: E402


def read_times(path: Path) -> np.ndarray:
    """Frame timestamps in microseconds, from an applanix ``*_times.txt``.

    The radar list ends with two bare labels, ``cart`` and ``mask``, naming the
    two derived radar products that sit beside the raw sweeps in the bucket, so
    anything that is not a frame name is skipped rather than parsed.
    """
    if not path.is_file():
        return np.zeros(0, dtype=np.int64)
    stamps = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stem = Path(line.strip()).stem
            if stem.isdigit():
                stamps.append(int(stem))
    return np.array(stamps, dtype=np.int64)


def stride_indices(count: int, stride: int, max_frames: int) -> np.ndarray:
    """The indices a download takes: every ``stride``-th key, up to ``max_frames``.

    Mirrors the selection in ``scripts/download_boreas.ps1``, so these numbers
    describe the frames that are actually on disk.
    """
    indices = np.arange(0, count, max(1, stride), dtype=np.int64)
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


def travelled_distance_m(poses) -> np.ndarray:
    """Cumulative distance along the route, in metres, one entry per pose row."""
    easting = np.asarray(poses.values["easting"], dtype=float)
    northing = np.asarray(poses.values["northing"], dtype=float)
    steps = np.hypot(np.diff(easting), np.diff(northing))
    return np.concatenate([[0.0], np.cumsum(steps)])


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--strides", default="20,200",
                        help="comma-separated strides to compare")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="frames per sensor the download takes")
    args = parser.parse_args()
    strides = [int(value) for value in args.strides.split(",") if value.strip()]

    cfg = load_config()
    data_root = Path(cfg["paths.data_raw"]) / "boreas"
    sequences = discover_sequences(data_root, data_root / "boreas-objects-v1")
    if not sequences:
        print(f"no sequences under {data_root}; download some first")
        return 1

    for sequence in sequences:
        applanix = sequence.root / "applanix"
        times = {kind: read_times(applanix / f"{kind}_times.txt")
                 for kind in ("lidar", "camera", "radar")}
        if not times["lidar"].size:
            print(f"\n{sequence.name}: no applanix time lists, skipped")
            continue

        drive = (times["lidar"][-1] - times["lidar"][0]) / 1e6
        print(f"\n{sequence.name}")
        print(f"  recording : {drive:.0f} s, "
              f"lidar {times['lidar'].size} frames, camera {times['camera'].size}, "
              f"radar {times['radar'].size}")

        try:
            travelled = travelled_distance_m(sequence.poses("lidar"))
        except (OSError, ValueError, KeyError):
            travelled = np.zeros(0)

        for stride in strides:
            picks = {kind: stride_indices(stamps.size, stride, args.max_frames)
                     for kind, stamps in times.items()}
            span = {kind: float(times[kind][indices][-1] - times[kind][indices][0]) / 1e6
                    for kind, indices in picks.items()}
            # A frame is usable for a multi-modal task only where every sensor
            # has one, so the shortest span decides how many of them qualify.
            origin = times["lidar"][picks["lidar"][0]]
            horizon = origin + int(min(span.values()) * 1e6)
            usable = int(np.sum(times["lidar"][picks["lidar"]] <= horizon))
            print(f"  stride {stride:>3}, {args.max_frames} frames per sensor:")
            for kind in ("lidar", "camera", "radar"):
                print(f"      {kind:<7} covers {span[kind]:>7.1f} s "
                      f"({100.0 * span[kind] / drive:>5.1f}% of the drive)")
            print(f"      usable multi-modal frames: {usable}")
            if travelled.size > int(picks["lidar"][-1]):
                first, last = picks["lidar"][0], picks["lidar"][-1]
                print(f"      route travelled by those frames: "
                      f"{travelled[last] - travelled[first]:.0f} m "
                      f"of {travelled[-1]:.0f} m in the recording")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())