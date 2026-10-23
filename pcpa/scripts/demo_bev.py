"""Smoke test for BEV rasterisation and the occupancy network.

Run from the project root::

    .\\.venv\\Scripts\\python.exe scripts\\demo_bev.py

Checks the rasterisers against the real Boreas frames that have been
downloaded, then trains the network for a few epochs on a synthetic occupancy
problem. The aim is to prove the machinery runs and that training reduces the
loss --- not to produce a result.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config                              # noqa: E402
from src.data.boreas import (                                          # noqa: E402
    BoreasSequence,
    discover_sequences,
    load_lidar_frame,
    load_radar_frame,
)
from src.degrade import lidar as lid                                  # noqa: E402
from src.degrade.severity import SeverityLadder                       # noqa: E402
from src.models.bev import (                                          # noqa: E402
    BevGrid,
    lidar_to_bev,
    occupancy_reference,
    radar_to_bev,
)
from src.models.network import (                                       # noqa: E402
    TrainConfig,
    build_model,
    evaluate,
    fit,
    input_channels,
)

FRAMES = 24
SEVERITY = 3


def synthetic_scene(grid: BevGrid, rng: np.random.Generator) -> np.ndarray:
    """A corridor with a few rectangular obstacles, as a point cloud.

    Used when no real frames are present, so the script runs before and after
    the download.
    """
    parts = []
    ground = 40000
    x = rng.uniform(-grid.extent_m, grid.extent_m, ground)
    y = rng.uniform(-grid.extent_m, grid.extent_m, ground)
    parts.append(np.stack([x, y, rng.normal(0.0, 0.02, ground),
                           rng.uniform(0.0, 200.0, ground),
                           np.zeros(ground), np.zeros(ground)], axis=1))
    wall = 6000
    wall_x = rng.uniform(-grid.extent_m, grid.extent_m, wall)
    for sign in (-1.0, 1.0):
        parts.append(np.stack([wall_x, np.full(wall, sign * 6.0),
                               rng.normal(0.4, 0.05, wall),
                               rng.uniform(0.0, 200.0, wall),
                               np.zeros(wall), np.zeros(wall)], axis=1))
    for _ in range(6):
        centre_x, centre_y = rng.uniform(-30.0, 30.0), rng.uniform(-20.0, 20.0)
        block = 400
        parts.append(np.stack([rng.normal(centre_x, 1.2, block),
                               rng.normal(centre_y, 1.2, block),
                               rng.normal(0.8, 0.3, block),
                               rng.uniform(0.0, 200.0, block),
                               np.zeros(block), np.zeros(block)], axis=1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def main() -> int:
    cfg = load_config()
    ladder = SeverityLadder.from_config(cfg)
    grid = BevGrid()
    rng = np.random.default_rng(0)
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        print(f"[{'OK  ' if ok else 'FAIL'}] {name:<36} {detail}")
        if not ok:
            failures += 1

    print(f"BEV grid: {grid.cells}x{grid.cells} at {grid.resolution_m:.3f} m/cell "
          f"over +/-{grid.extent_m:.0f} m\n")

    data_root = Path(cfg["paths.data_raw"]) / "boreas"
    bundle = data_root / "boreas-objects-v1"
    sequences = discover_sequences(data_root, bundle)

    cloud = None
    radar_frame = None
    for sequence in sequences:
        if sequence.frames("lidar"):
            cloud = load_lidar_frame(sequence.frames("lidar")[0])
            print(f"lidar source : {sequence.frames('lidar')[0].name} ({sequence.name})")
            break
    if (bundle / "labels_detection").is_dir():
        bundle_sequence = BoreasSequence(bundle)
        if cloud is None and bundle_sequence.frames("lidar"):
            cloud = load_lidar_frame(bundle_sequence.frames("lidar")[0])
        if bundle_sequence.frames("radar"):
            radar_frame = load_radar_frame(bundle_sequence.frames("radar")[0])
    if cloud is None:
        cloud = synthetic_scene(grid, rng)
        print("lidar source : synthetic")
    else:
        print(f"               {cloud.shape[0]:,} points")

    channels = lidar_to_bev(cloud, grid)
    check("lidar rasterises to BEV", channels.shape == (4, *grid.shape),
          f"{channels.shape}, {int(channels[0].sum()):,} occupied cells, "
          f"max height {float(channels[1].max()):.2f} m")

    reference = occupancy_reference(cloud, grid)
    check("occupancy reference", reference.shape == (1, *grid.shape),
          f"{int(reference.sum()):,} occupied cells")

    if radar_frame is not None:
        radar_bev = radar_to_bev(radar_frame, grid)
        check("radar rasterises to BEV", radar_bev.shape == (1, *grid.shape),
              f"polar {radar_frame.shape} -> {radar_bev.shape}, "
              f"max {float(radar_bev.max()):.3f}")
    else:
        print("[    ] no radar frame downloaded, skipping the radar rasteriser")

    print()
    params = ladder.at(SEVERITY).as_dict()
    inputs, targets = [], []
    for _ in range(FRAMES):
        scene = synthetic_scene(grid, rng)
        targets.append(occupancy_reference(scene, grid))
        degraded = lid.degrade(scene, lid.LidarDegradation(
            extinction_beta_per_m=params["lidar_beta_per_m"],
            spurious_return_rate=params["lidar_spurious_rate"],
            beam_dilation_rad=params["lidar_beam_dilation_rad"],
            range_noise_frac=params["lidar_range_noise_frac"]), rng)
        inputs.append(lidar_to_bev(degraded, grid))

    x = np.stack(inputs)
    y = np.stack(targets)
    split = FRAMES - 6
    train_x, train_y = x[:split], y[:split]
    val_x, val_y = x[split:], y[split:]
    print(f"synthetic task: {FRAMES} frames, severity s={SEVERITY}, "
          f"{split} train / {FRAMES - split} validation")

    device = "cuda"
    config = TrainConfig(epochs=6, batch_size=4, learning_rate=2e-3, device=device)
    model = build_model(("lidar",), base_channels=16, device=device)
    parameters = sum(p.numel() for p in model.parameters())
    check("model builds", parameters > 0,
          f"lidar only -> {input_channels(('lidar',))} input channels, "
          f"{parameters:,} parameters on {device}")

    start = time.time()
    history = fit(model, train_x, train_y, config, validation=(val_x, val_y))
    elapsed = time.time() - start
    for record in history:
        print(f"    epoch {int(record['epoch'])}  train_loss={record['train_loss']:.4f}  "
              f"val_iou={record.get('val_iou', float('nan')):.4f}")
    check("training reduces the loss",
          history[-1]["train_loss"] < history[0]["train_loss"],
          f"{history[0]['train_loss']:.4f} -> {history[-1]['train_loss']:.4f} "
          f"in {elapsed:.1f} s on {device}")
    check("held-out IoU beats chance", evaluate(model, val_x, val_y, config) > 0.2,
          f"IoU={evaluate(model, val_x, val_y, config):.4f}")

    print()
    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("BEV rasterisation and the occupancy network both work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

