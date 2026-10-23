"""Per-modality retention on the ladder, measured rather than illustrative.

    .\\.venv\\Scripts\\python.exe scripts\\per_modality_retention.py

Trains one model per modality on levels 0--2 and evaluates it on all five, so
that "how much does each sensor retain as severity rises" is answered by
measurement. The earlier revision answered it with a hand-drawn curve, which is
exactly the kind of figure that cannot survive review.

The protocol is the one the rest of the paper reports --- 40 frames per sequence,
eight epochs, AdamW, the same split and the same rasterisation --- so these
numbers sit in the same table as the fusion results instead of in a parallel
study. Output goes to ``data/manifest/retention_<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_a2_ablation as a2                                             # noqa: E402
import run_fusion_study as fusion                                        # noqa: E402

from src.common.config import load_config                                # noqa: E402
from src.degrade.severity import SeverityLadder                          # noqa: E402
from src.models.bev import BevGrid, occupancy_reference                  # noqa: E402
from src.models.network import (BevOccupancyNet, TrainConfig,            # noqa: E402
                                evaluate, fit, input_channels)

#: The single modalities, plus the all-modality stack as the reference row.
SUBSETS = (("camera",), ("lidar",), ("radar",), ("camera", "lidar", "radar"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-sequence", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--train-levels", default="0,1,2")
    parser.add_argument("--degradation-seed", type=int, default=0,
                        help="seed for the degradation operators, separate from the "
                             "training seed so every run sees identical inputs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    grid = BevGrid()
    ladder = SeverityLadder.from_config(cfg)
    camera_shape = (300, 400)
    camera_calib = a2.camera_calibration_or_default(cfg, camera_shape)
    train_levels = tuple(int(value) for value in str(args.train_levels).split(","))

    scenes, cameras, radars, _, sequence_names = a2.build_real_frames(
        cfg, camera_shape, args.frames_per_sequence, args.stride)
    if len(scenes) < 6:
        print("not enough aligned frames")
        return 1
    total = len(scenes)
    targets = np.stack([occupancy_reference(scene, grid) for scene in scenes])
    train_index, test_index = fusion.stratify_by_sequence(sequence_names,
                                                          fusion.TRAIN_FRACTION)
    train_count = int(train_index.size)
    test = test_index
    print(f"frames   : {total} ({train_count} train / {test_index.size} test), "
          f"split per sequence")
    print(f"protocol : {args.epochs} epochs, base channels {args.base_channels}, "
          f"trained on levels {train_levels}\n")

    channels = {level: fusion.channels_for_level(
        cfg, ladder, scenes, cameras, radars, camera_calib, grid, level,
        args.degradation_seed, sequence_names, args.frames_per_sequence, args.stride)
        for level in fusion.TEST_LEVELS}
    config = TrainConfig(epochs=args.epochs, device="cuda", seed=args.seed)

    started = time.time()
    results: dict[str, list[float]] = {}
    for modalities in SUBSETS:
        train_x = np.concatenate(
            [fusion.stacked(channels, level, modalities)[train_index]
             for level in train_levels], axis=0)
        train_y = np.concatenate([targets[train_index]] * len(train_levels), axis=0)
        model = BevOccupancyNet(input_channels(modalities), args.base_channels)
        name = "+".join(modalities)
        print(f"{name}")
        fit(model, train_x, train_y, config)
        results[name] = [
            evaluate(model, fusion.stacked(channels, level, modalities)[test],
                     targets[test]) for level in fusion.TEST_LEVELS]
        print("  " + "  ".join(f"s={level}:{value:.4f}"
                               for level, value in zip(fusion.TEST_LEVELS,
                                                       results[name])))

    print("\nBEV IoU per modality subset (higher is better)")
    print(f"{'subset':<22}" + "".join(f"{'s=' + str(level):>9}"
                                      for level in fusion.TEST_LEVELS))
    for name, values in results.items():
        print(f"{name:<22}" + "".join(f"{value:>9.4f}" for value in values))

    destination = ROOT / "data" / "manifest"
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / f"retention_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"args": vars(args), "frames": {"total": total,
                                                              "train": train_count},
                               "iou": results,
                               "elapsed_seconds": round(time.time() - started, 1)},
                              indent=2, default=float), encoding="utf-8")
    print(f"\nresult written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
