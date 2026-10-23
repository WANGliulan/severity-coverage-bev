"""Move aside the frames that the aligned download did not ask for.

The downloader picks a LiDAR sample first and then the camera and radar frames
nearest those timestamps, so that the three modalities describe the same
moments. An earlier plan sampled every sensor by its own key index instead, and
its files are still on disk: they have the right names, they are complete, and
they pair with nothing --- measured on boreas-2020-11-26-13-58, 79 of its 114
LiDAR frames had no camera partner, which is also why demo_boreas.py reported
"camera matched within tolerance: worst |dt| = 474.3 ms".

This moves those files to ``data/_stale/<sequence>/`` rather than deleting them.
It is meant to run unattended, and a wrong guess about what is stale should cost
disk space rather than data.

    .\\.venv\\Scripts\\python.exe scripts\\prune_unaligned.py --dry-run
    .\\.venv\\Scripts\\python.exe scripts\\prune_unaligned.py
"""

from __future__ import annotations

import argparse
import bisect
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config                             # noqa: E402
from src.data.boreas import discover_sequences                        # noqa: E402

#: What each frame modality is stored as, which is also what decides which files
#: this script is allowed to touch.
FRAME_SUFFIX = {"lidar": ".bin", "camera": ".png", "radar": ".png"}

#: How far a partner frame may sit from a LiDAR sweep and still be considered
#: the same moment. Kept identical to MATCH_TOLERANCE_US in
#: scripts/run_a2_ablation.py --- if the two disagreed, this script would keep
#: frames that the ablation then rejects. The sensors do not share a rate: the
#: camera runs at ~20 Hz (neighbours 50 ms apart) and the radar at 4 Hz (250 ms).
TOLERANCE_US = {"camera": 60_000, "radar": 190_000}


def read_times(path: Path) -> list[int]:
    """Timestamps of every frame of one sensor, from an applanix list.

    The radar list ends with the bare labels ``cart`` and ``mask``, so anything
    that is not a frame name is skipped.
    """
    if not path.is_file():
        return []
    stamps = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stem = Path(line.strip()).stem
            if stem.isdigit():
                stamps.append(int(stem))
    return stamps


def nearest_index(stamps: list[int], target: int) -> int:
    """Index of the stamp closest to ``target``; ``stamps`` must be sorted."""
    position = bisect.bisect_left(stamps, target)
    if position <= 0:
        return 0
    if position >= len(stamps):
        return len(stamps) - 1
    before, after = stamps[position - 1], stamps[position]
    return position - 1 if target - before <= after - target else position


def aligned_names(sequence_dir: Path, stride: int, max_frames: int) -> dict[str, set[str]]:
    """The file names the downloader's aligned plan asks for, per modality.

    Mirrors the selection in scripts/download_boreas.ps1: every ``stride``-th
    LiDAR frame up to ``max_frames``, then the camera and radar frames nearest
    those timestamps.
    """
    applanix = sequence_dir / "applanix"
    times = {kind: read_times(applanix / f"{kind}_times.txt") for kind in FRAME_SUFFIX}
    if not times["lidar"]:
        return {}

    sample = times["lidar"][::stride][:max_frames]
    keep = {"lidar": {f"{stamp}.bin" for stamp in sample}}
    for kind in ("camera", "radar"):
        stamps = times[kind]
        if not stamps:
            keep[kind] = set()
            continue
        keep[kind] = {f"{stamps[nearest_index(stamps, stamp)]}.png" for stamp in sample}
    return keep


def keep_by_tolerance(sequence_dir: Path) -> dict[str, set[str]]:
    """Frames that take part in at least one time-coherent triple.

    Needed because only the first downloaded sequence ships ``*_times.txt``;
    without the full frame list there is no way to reproduce the downloader's
    index-based selection. There is also no need to: a frame that pairs with
    nothing is useless to a multi-modal task whether or not it was planned, and
    a frame that does pair is worth keeping wherever it came from.

    Conservative by construction --- only frames that join no triple at all are
    left out, so nothing usable can be moved aside.
    """
    stamps, names = {}, {}
    for kind, suffix in FRAME_SUFFIX.items():
        folder = sequence_dir / kind
        if not folder.is_dir():
            stamps[kind], names[kind] = [], []
            continue
        names[kind] = sorted(path.name for path in folder.glob(f"*{suffix}"))
        stamps[kind] = [int(Path(name).stem) for name in names[kind]]

    keep = {kind: set() for kind in FRAME_SUFFIX}
    for index, stamp in enumerate(stamps["lidar"]):
        partners = {}
        for kind in ("camera", "radar"):
            if not stamps[kind]:
                partners = {}
                break
            hit = nearest_index(stamps[kind], stamp)
            if abs(stamps[kind][hit] - stamp) > TOLERANCE_US[kind]:
                partners = {}
                break
            partners[kind] = hit
        if len(partners) < 2:
            continue
        keep["lidar"].add(names["lidar"][index])
        for kind, hit in partners.items():
            keep[kind].add(names[kind][hit])
    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--stride", type=int, default=300,
                        help="must match the -Stride the download used")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="must match the -MaxFrames the download used")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would move, and move nothing")
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(cfg["paths.data_raw"]) / "boreas"
    stale_root = ROOT / "data" / "_stale"
    sequences = discover_sequences(data_root, data_root / "boreas-objects-v1")
    if not sequences:
        print(f"no sequences under {data_root}")
        return 1

    moved_total, size_total = 0, 0.0
    for sequence in sequences:
        keep = aligned_names(sequence.root, args.stride, args.max_frames)
        if keep:
            method = f"planned, stride {args.stride}"
        else:
            keep = keep_by_tolerance(sequence.root)
            method = "time-coherent only"
        if not any(keep.values()):
            print(f"{sequence.name}: no usable frames found, left untouched")
            continue

        wanted_count = sum(len(names) for names in keep.values())
        moved, size = 0, 0.0
        for kind, suffix in FRAME_SUFFIX.items():
            folder = sequence.root / kind
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob(f"*{suffix}")):
                if path.name in keep[kind]:
                    continue
                moved += 1
                size += path.stat().st_size
                if not args.dry_run:
                    destination = stale_root / sequence.name / kind / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(destination))

        moved_total += moved
        size_total += size
        verb = "would move" if args.dry_run else "moved"
        print(f"{sequence.name}: keeps {wanted_count} frame(s) [{method}], "
              f"{verb} {moved} file(s), {size / 1e6:.0f} MB")

    action = "would move" if args.dry_run else "moved"
    print(f"\n{action} {moved_total} file(s), {size_total / 1e6:.0f} MB in total")
    if args.dry_run and moved_total:
        print("re-run without --dry-run to move them")
    elif moved_total:
        print(f"they are under {stale_root}, so a wrong call can be undone by "
              f"moving them back")
    else:
        print("nothing to move: the directories already match the aligned plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())