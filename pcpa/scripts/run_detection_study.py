"""Object detection on the labelled subset, under the same degradation ladder.

    .\\.venv\\Scripts\\python.exe scripts\\run_detection_study.py

The occupancy study needs no annotation, so it could be run on all fourteen
sequences. This one needs boxes, and boxes exist for a single sequence: the 4951
labelled frames of ``boreas-objects-v1``, of which 279 also carry the LiDAR,
camera and radar frames the protocol needs. That is a smaller experiment than the
occupancy tables, and it is stated as such rather than glossed over.

Everything else is held fixed --- the same five-level ladder, the same BEV grid,
the same rasterisation, the same per-sequence split --- so a difference between
two fusion structures here is a difference in the trunk. The metric is the one
detection benchmarks use: per-class AP pooled over the whole split, matched
greedily at an IoU threshold with all points interpolated, reported per severity
level, because a single average would hide the effect the paper is about.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_a2_ablation as a2                                          # noqa: E402
import run_fusion_study as fusion                                     # noqa: E402

from src.common.config import load_config                             # noqa: E402
from src.data.boreas import (BoreasSequence, frame_timestamp,          # noqa: E402
                             load_camera_frame, load_lidar_frame,
                             load_radar_frame)
from src.degrade.severity import SeverityLadder                       # noqa: E402
from src.eval.detection import (Box, average_precision,              # noqa: E402
                                mean_average_precision)
from src.models.bev import BevGrid, occupancy_reference               # noqa: E402
from src.models.detection import (DetectionConfig, Detector,          # noqa: E402
                                  decode_detections, detection_loss,
                                  detection_targets)
from src.models.gated import GatedFusionNet                           # noqa: E402
from src.models.network import (BevOccupancyNet, TrainConfig,         # noqa: E402
                                input_channels, occupancy_loss)

MANIFEST = ROOT / "data" / "manifest"
MODALITIES = fusion.MODALITIES
TRAIN_LEVELS = (0, 1, 2)
TEST_LEVELS = (0, 1, 2, 3, 4)
BUNDLE = "boreas-objects-v1"


def _format(value: float | None) -> str:
    """A metric that may not exist, printed without pretending it does.

    A class with no ground truth in the split has no AP, and showing 0.000 for it
    would read as a failure of the model rather than an absence in the labels.
    """
    return "n/a" if value is None else f"{value:.4f}"


def boxes_from_labels(labels) -> list[Box]:
    """Label records to BEV boxes.

    The three dimensions in a Boreas label file are length, width and height, in
    that order: the sample's car reads 4.714 / 2.123 / 1.681, a pedestrian 0.66 /
    0.73 / 1.40 and a bus 11.2 / 2.94 / 3.98. The reader names the *first* field
    ``height_m``, so the field whose name reads like a height is the length. Using
    the names instead of the values would make every car box 4.7 m wide and 1.7 m
    long, which does not look like a crash --- it looks like a low IoU.
    """
    return [Box(x=label.x_m, y=label.y_m, length=label.height_m,
                width=label.width_m, heading=label.rotation_y_rad,
                score=1.0, label=label.category)
            for label in labels]


def labelled_frames(cfg, limit: int, stride: int):
    """Frames with a label *and* all three modalities, aligned as the study requires.

    The alignment rule is the one ``run_a2_ablation.build_real_frames`` applies, so
    that "one frame" means the same thing in both studies. Frames whose label
    exists but whose sensors do not are counted rather than dropped silently.
    """
    bundle = Path(cfg["paths.data_raw"]) / "boreas" / BUNDLE
    sequence = BoreasSequence(bundle)
    labelled = {int(value) for value in sequence.label_timestamps()}
    scenes, cameras, radars, boxes, timestamps = [], [], [], [], []
    rejected = {"unlabelled": 0, "misaligned": 0, "unreadable": 0}
    for path in sequence.frames("lidar")[::stride]:
        timestamp = frame_timestamp(path)
        if timestamp not in labelled:
            rejected["unlabelled"] += 1
            continue
        camera_path = sequence.nearest("camera", timestamp)
        radar_path = sequence.nearest("radar", timestamp)
        if camera_path is None or radar_path is None:
            rejected["misaligned"] += 1
            continue
        offsets = {"camera": abs(frame_timestamp(camera_path) - timestamp),
                   "radar": abs(frame_timestamp(radar_path) - timestamp)}
        if any(offsets[kind] > a2.MATCH_TOLERANCE_US[kind] for kind in offsets):
            rejected["misaligned"] += 1
            continue
        try:
            cloud = load_lidar_frame(path)
            image = load_camera_frame(camera_path)
            radar = load_radar_frame(radar_path)
        except (ValueError, OSError):
            rejected["unreadable"] += 1
            continue
        scenes.append(cloud)
        cameras.append(image)
        radars.append(radar)
        boxes.append(boxes_from_labels(sequence.labels_for(timestamp)))
        timestamps.append(timestamp)
        if limit and len(scenes) >= limit:
            break
    return sequence, scenes, cameras, radars, boxes, timestamps, rejected


def encode_targets(boxes_per_frame, grid: BevGrid,
                   config: DetectionConfig) -> dict[str, np.ndarray]:
    """Stack the per-frame detection targets into one tensor per head output."""
    encoded = [detection_targets(boxes, grid, config) for boxes in boxes_per_frame]
    return {"heatmap": np.stack([item.heatmap for item in encoded]),
            "size": np.stack([item.parameters[0:2] for item in encoded]),
            "offset": np.stack([item.parameters[2:4] for item in encoded]),
            "heading": np.stack([item.parameters[4:6] for item in encoded]),
            "mask": np.stack([item.mask for item in encoded])}


def train_detector(detector: Detector, inputs, occupancy, encoded, levels,
                   config: TrainConfig) -> list[dict[str, float]]:
    """Joint occupancy and detection loss, AdamW, the schedule used everywhere else.

    The occupancy term is not decoration: without it the trunk would be trained for
    boxes only, and the model would not be the same object as the one the occupancy
    tables describe. The severity auxiliary loss is kept for the gated trunk, again
    for the same reason --- it is what anchors the reliability table.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    detector = detector.to(device)
    torch.manual_seed(config.seed)
    x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(occupancy), dtype=torch.float32)
    tensors = {name: torch.as_tensor(np.asarray(value), dtype=torch.float32)
               for name, value in encoded.items()}
    level_values = np.asarray(levels, dtype=np.int64)
    if level_values.ndim == 1:
        # One severity for the whole frame: every modality was degraded alike.
        level_values = np.repeat(level_values[:, None], len(MODALITIES), axis=1)
    level_tensor = torch.as_tensor(level_values, dtype=torch.long)
    optimiser = torch.optim.AdamW(detector.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        detector.train()
        order = torch.randperm(x.shape[0])
        running = 0.0
        for start in range(0, x.shape[0], config.batch_size):
            index = order[start:start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            logits, outputs, severity = detector(x[index].to(device))
            loss = occupancy_loss(logits, y[index].to(device))
            loss = loss + detection_loss(
                outputs, {name: value[index].to(device)
                          for name, value in tensors.items()})
            if severity is not None:
                auxiliary = sum(
                    nn.functional.cross_entropy(
                        severity[:, modality, :],
                        level_tensor[index][:, modality].to(device))
                    for modality in range(severity.shape[1])) / severity.shape[1]
                loss = loss + config.auxiliary_weight * auxiliary
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * index.numel()
        history.append({"epoch": float(epoch),
                        "loss": running / max(x.shape[0], 1)})
        print(f"    epoch {epoch + 1}/{config.epochs}  "
              f"loss {history[-1]['loss']:.4f}", flush=True)
    return history


@torch.no_grad()
def predict(detector: Detector, inputs, grid: BevGrid, config: DetectionConfig,
            batch_size: int = 8) -> list[list[Box]]:
    """Decoded boxes for every frame, in the order the frames were given."""
    device = next(detector.parameters()).device
    detector.eval()
    out: list[list[Box]] = []
    for start in range(0, inputs.shape[0], batch_size):
        block = torch.as_tensor(np.asarray(inputs[start:start + batch_size]),
                                dtype=torch.float32, device=device)
        _, outputs, _ = detector(block)
        decoded = decode_detections({name: value.cpu()
                                     for name, value in outputs.items()}, grid, config)
        out.extend(decoded)
    return out


@torch.no_grad()
def occupancy_iou(detector: Detector, inputs, targets, batch_size: int = 8) -> float:
    """Occupancy IoU over a split, pooled as ``bev_iou`` pools it."""
    device = next(detector.parameters()).device
    detector.eval()
    intersection = union = 0.0
    for start in range(0, inputs.shape[0], batch_size):
        block = torch.as_tensor(np.asarray(inputs[start:start + batch_size]),
                                dtype=torch.float32, device=device)
        logits, _, _ = detector(block)
        reference = torch.as_tensor(np.asarray(targets[start:start + batch_size]),
                                    dtype=torch.float32, device=device)
        prediction = (torch.sigmoid(logits) > 0.5).float()
        intersection += float((prediction * reference).sum())
        union += float(prediction.sum() + reference.sum())
    return intersection / max(union - intersection, 1e-7)


def score(predictions: list[list[Box]], truth: list[list[Box]],
          iou_threshold: float) -> dict[str, float | None]:
    """Per-class AP plus the mAP at one IoU threshold."""
    summary: dict[str, float | None] = {
        f"ap_{name}": value for name, value in
        average_precision(predictions, truth, iou_threshold=iou_threshold).items()}
    summary["map"] = mean_average_precision(predictions, truth,
                                            iou_threshold=iou_threshold)
    return summary


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters, including the detection head."""
    return int(sum(parameter.numel() for parameter in model.parameters()
                   if parameter.requires_grad))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames", type=int, default=0,
                        help="cap on labelled frames; 0 uses every usable one")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--train-levels", default="0,1,2",
                        help="severities mixed into training; '0,1,2,3,4' measures "
                             "the full-coverage upper bound, as it does for occupancy")
    parser.add_argument("--degradation-seed", type=int, default=0,
                        help="seed for the degradation operators, separate from the "
                             "training seed so every run sees identical inputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="a dozen frames and one epoch, to check the plumbing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not fusion.acquire_lock():
        return 1
    atexit.register(fusion.release_lock)
    if args.smoke:
        # Only the frame count is capped; the epoch count stays as asked, so a
        # smoke run can be long enough to show the metrics move off zero.
        args.frames = args.frames or 12
    train_levels = tuple(int(value) for value in str(args.train_levels).split(","))
    started = time.time()

    cfg = load_config()
    grid = BevGrid()
    ladder = SeverityLadder.from_config(cfg)
    camera_shape = (300, 400)
    camera_calib = a2.camera_calibration_or_default(cfg, camera_shape)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detection_config = DetectionConfig()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"source   : the labelled subset only, "
          f"{args.frames if args.frames else 'all usable'} frames, stride {args.stride}")
    print(f"device   : {device}")
    print(f"severity : train on {train_levels}, evaluate on {TEST_LEVELS}")
    print(f"classes  : {detection_config.classes} for the mean, "
          f"Misc reported alongside\n")

    sequence, scenes, cameras, radars, boxes, _, rejected = labelled_frames(
        cfg, args.frames, args.stride)
    if len(scenes) < 8:
        print(f"not enough labelled frames ({len(scenes)}); check the bundle download")
        return 1
    per_frame = np.array([len(frame) for frame in boxes])
    total_objects = int(per_frame.sum())
    in_grid = sum(1 for frame in boxes for box in frame
                  if abs(box.x) <= grid.extent_m and abs(box.y) <= grid.extent_m)
    print(f"sequence : {sequence.name}, {len(scenes)} usable frames, "
          f"rejected {rejected}")
    print(f"objects  : {total_objects} boxes, mean {per_frame.mean():.1f} per frame, "
          f"max {per_frame.max()}")
    print(f"reach    : {in_grid}/{total_objects} boxes inside the "
          f"+/-{grid.extent_m:.0f} m grid ({100 * in_grid / max(total_objects, 1):.1f}%)")

    targets = np.stack([occupancy_reference(scene, grid) for scene in scenes])
    encoded = encode_targets(boxes, grid, detection_config)
    names = np.full(len(scenes), sequence.name)
    train_index, test_index = fusion.stratify_by_sequence(names, fusion.TRAIN_FRACTION)
    truth = [boxes[index] for index in test_index]
    print(f"\nsplit    : {train_index.size} train / {test_index.size} test frames, "
          f"per sequence")

    print("\nbuilding the degraded BEV stacks")
    channels = {level: fusion.channels_for_level(
        cfg, ladder, scenes, cameras, radars, camera_calib, grid, level,
        args.degradation_seed, names, len(scenes), args.stride)
        for level in TEST_LEVELS}

    train_x = np.concatenate([fusion.stacked(channels, level, MODALITIES)[train_index]
                              for level in train_levels], axis=0)
    train_y = np.concatenate([targets[train_index]] * len(train_levels), axis=0)
    train_encoded = {name: np.concatenate([value[train_index]] * len(train_levels),
                                          axis=0)
                     for name, value in encoded.items()}
    level_labels = np.concatenate([np.full(train_index.size, level, dtype=np.int64)
                                   for level in train_levels])
    config = TrainConfig(epochs=args.epochs, device=device, seed=args.seed)
    print(f"training : {train_x.shape[0]} samples of {train_x.shape[1]} channels, "
          f"{args.epochs} epoch(s)")

    results: dict[str, dict] = {}
    # The head reads the *decoder* features, whose width is the trunk's
    # base_channels rather than the number of input channels; the two trunks share
    # that convention, which is what lets one head sit on either of them.
    for label, trunk in (
            ("concat", BevOccupancyNet(input_channels(MODALITIES), args.base_channels)),
            ("gate", GatedFusionNet(MODALITIES, num_levels=len(TEST_LEVELS),
                                    base_channels=args.base_channels))):
        detector = Detector(trunk, args.base_channels, detection_config)
        print(f"\n{label:<8} ({parameter_count(detector)} parameters)")
        history = train_detector(detector, train_x, train_y, train_encoded,
                                 level_labels, config)
        per_level: dict[str, dict] = {}
        for level in TEST_LEVELS:
            inputs = fusion.stacked(channels, level, MODALITIES)[test_index]
            predictions = predict(detector, inputs, grid, detection_config)
            strict = score(predictions, truth, 0.5)
            loose = score(predictions, truth, 0.3)
            entry = {
                "iou": occupancy_iou(detector, inputs, targets[test_index]),
                "map_at_0.5": strict["map"],
                "map_at_0.3": loose["map"],
                "ap_at_0.5": {name: value for name, value in strict.items()
                              if name.startswith("ap_")},
                "detections_per_frame": float(np.mean([len(frame)
                                                       for frame in predictions])),
            }
            per_level[f"s={level}"] = entry
            print(f"  s={level}: occupancy IoU {entry['iou']:.4f}  "
                  f"mAP@0.5 {_format(entry['map_at_0.5'])}  "
                  f"mAP@0.3 {_format(entry['map_at_0.3'])}  "
                  f"boxes/frame {entry['detections_per_frame']:.1f}")
        results[label] = {"per_level": per_level,
                          "parameters": parameter_count(detector),
                          "history": history}

    print("\nmAP at BEV IoU 0.5 (higher is better)")
    header = f"{'configuration':<14}" + "".join(f"{'s=' + str(level):>10}"
                                                for level in TEST_LEVELS)
    print(header)
    print("-" * len(header))
    for label, payload in results.items():
        print(f"{label:<14}" + "".join(
            f"{_format(payload['per_level'][f's={level}']['map_at_0.5']):>10}"
            for level in TEST_LEVELS))
    worst = TEST_LEVELS[-1]
    print(f"\nper-class AP at s={worst}")
    for label, payload in results.items():
        ap = payload["per_level"][f"s={worst}"]["ap_at_0.5"]
        print(f"{label:<14}" + "  ".join(f"{name}={_format(value)}"
                                         for name, value in sorted(ap.items())))
    payload = {
        "args": vars(args),
        "train_levels": list(train_levels),
        "sequence": sequence.name,
        "frames": {"total": len(scenes), "train": int(train_index.size),
                   "test": int(test_index.size)},
        "objects": {"total": total_objects, "in_grid": int(in_grid)},
        "rejected": rejected,
        "detection": {"classes": list(detection_config.classes),
                      "min_score": detection_config.min_score,
                      "max_detections": detection_config.max_detections,
                      "grid_extent_m": grid.extent_m},
        "results": results,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    MANIFEST.mkdir(parents=True, exist_ok=True)
    path = MANIFEST / f"detection_{time.strftime('%Y%m%d-%H%M%S')}.json"
    if args.smoke:
        print("\nsmoke run: no record written")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nresult written to {path} | elapsed: {int(time.time() - started)} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


