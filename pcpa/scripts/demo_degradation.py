"""Severity sweep over the three degraded modalities.

Run from the project root::

    .\\.venv\\Scripts\\python.exe scripts\\demo_degradation.py

Uses the real Boreas recordings under ``data/raw/boreas`` when they are
present and falls back to synthetic inputs otherwise, so the script runs
before and after the download. It writes a montage to
``figures/degradation_grid.png`` and prints the statistics that the severity
figure of the paper is drawn from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from PIL import Image                                                # noqa: E402

from src.common.config import load_config                            # noqa: E402
from src.degrade import camera as cam                                # noqa: E402
from src.degrade import lidar as lid                                 # noqa: E402
from src.degrade import radar as rad                                 # noqa: E402
from src.degrade.severity import SeverityLadder                      # noqa: E402


def first_file(pattern: str) -> Path | None:
    """First matching file under data/raw/boreas, or None."""
    hits = sorted((ROOT / "data" / "raw" / "boreas").glob(pattern))
    return hits[0] if hits else None


def synthetic_camera(shape: tuple[int, int, int] = (180, 320, 3)) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    base = 0.35 + 0.35 * np.sin(xx / 18.0) * np.cos(yy / 26.0)
    image = np.stack([base, base * 0.95 + 0.05, base * 0.80], axis=-1)
    for cy, cx, radius in [(60, 90, 18), (120, 210, 26), (40, 260, 12)]:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
        image[mask] = np.array([0.90, 0.20, 0.15])
    return np.clip(image, 0.0, 1.0)


def synthetic_lidar(count: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    elevation = rng.uniform(-0.30, 0.10, count)
    distance = rng.uniform(2.0, 80.0, count)
    xyz = np.stack([distance * np.cos(elevation) * np.cos(azimuth),
                    distance * np.cos(elevation) * np.sin(azimuth),
                    distance * np.sin(elevation)], axis=1)
    return np.concatenate([xyz,
                           rng.uniform(0.0, 200.0, count)[:, None],
                           distance[:, None],
                           np.zeros((count, 1))], axis=1).astype(np.float32)


def synthetic_radar(shape: tuple[int, int] = (400, 300), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = (rng.random(shape) * 0.05).astype(np.float32)
    for azimuth, range_bin in [(60, 120), (150, 200), (300, 80)]:
        frame[azimuth:azimuth + 3, range_bin:range_bin + 6] = 0.9
    return frame


def main() -> int:
    cfg = load_config()
    ladder = SeverityLadder.from_config(cfg)
    rng = np.random.default_rng(0)

    camera_file = first_file("*/camera/*.png")
    radar_file = first_file("*/radar/*.png")
    lidar_file = first_file("*/lidar/*.bin")

    image = np.asarray(Image.open(camera_file).convert("RGB")) if camera_file else synthetic_camera()
    if radar_file:
        radar_frame = np.asarray(Image.open(radar_file))[:400, :300]
    else:
        radar_frame = synthetic_radar()
    if lidar_file:
        cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 6)
        if cloud.shape[0] > 40000:
            cloud = cloud[rng.choice(cloud.shape[0], 40000, replace=False)]
    else:
        cloud = synthetic_lidar()

    print(f"camera source : {camera_file or 'synthetic'}")
    print(f"radar  source : {radar_file or 'synthetic'}")
    print(f"lidar  source : {lidar_file or 'synthetic'}   ({cloud.shape[0]:,} points)")
    print()
    print(ladder.table())
    print()

    header = (f"{'s':>2}{'cam_mean':>10}{'cam_contrast':>14}{'cam_dev':>9}"
              f"{'lidar_int':>11}{'radar_floor':>13}")
    print(header)
    print("-" * len(header))

    # The camera is judged by how far each level moves the frame from the clean
    # one, not by its contrast. Once the read noise dominates the exposure loss
    # the noise raises the contrast again, so contrast stops being monotone even
    # though every level is strictly worse to look at. The deviation from the
    # clean frame is monotone by construction: scattering, streaks, exposure and
    # noise each add to it and none of them can subtract.
    clean_float = cam.to_float_image(image)
    camera_panels = []
    lidar_panels = []
    radar_panels = []
    camera_contrasts = []
    camera_deviations = []
    lidar_intensities = []
    radar_floors = []

    for level in ladder.levels:
        p = ladder.at(float(level))
        cam_out = cam.degrade(image, cam.CameraDegradation(
            extinction_beta_per_m=p.camera_beta_per_m,
            streak_density=p.camera_streak_density,
            exposure_loss=p.camera_exposure_loss,
            read_noise_std=p.camera_read_noise_std,
            scene_depth_m=float(cfg["degradation.camera.scene_depth_m"]),
            atmospheric_light=float(cfg["degradation.camera.atmospheric_light"])), rng)
        lid_out = lid.degrade(cloud, lid.LidarDegradation(
            extinction_beta_per_m=p.lidar_beta_per_m,
            spurious_return_rate=p.lidar_spurious_rate,
            beam_dilation_rad=p.lidar_beam_dilation_rad,
            range_noise_frac=p.lidar_range_noise_frac,
            max_range_m=float(cfg["degradation.lidar.max_range_m"])), rng)
        rad_out = rad.degrade(radar_frame, rad.RadarDegradation(
            clutter_to_signal_db=p.radar_clutter_db,
            angular_smear_bins=p.radar_angular_smear_bins,
            looks=p.radar_looks,
            dynamic_range_db=float(cfg["degradation.radar.dynamic_range_db"])), rng)

        cam_float = cam.to_float_image(cam_out)
        lid_stats = lid.summary(lid_out)
        rad_stats = rad.summary(rad_out)
        camera_panels.append(cam_float)
        lidar_panels.append(lid_out)
        radar_panels.append(rad_out)
        camera_contrasts.append(float(cam_float.std()))
        camera_deviations.append(float(np.sqrt(np.mean((cam_float - clean_float) ** 2))))
        lidar_intensities.append(lid_stats["mean_intensity"])
        radar_floors.append(rad_stats["floor"])
        print(f"{int(level):>2}{cam_float.mean():>10.4f}{cam_float.std():>14.4f}"
              f"{camera_deviations[-1]:>9.4f}"
              f"{lid_stats['mean_intensity']:>11.2f}{rad_stats['floor']:>13.4f}")

    levels = ladder.levels.size
    fig, axes = plt.subplots(3, levels, figsize=(2.2 * levels, 6.6))
    for column, level in enumerate(ladder.levels):
        axes[0, column].imshow(np.clip(camera_panels[column], 0.0, 1.0))
        axes[0, column].set_title(f"s={int(level)}", fontsize=9)
        pts = lidar_panels[column]
        axes[1, column].scatter(pts[:, 0], pts[:, 1], s=0.2,
                                c=pts[:, 3] if pts.shape[1] > 3 else None,
                                cmap="viridis", vmin=0.0, vmax=200.0)
        axes[1, column].set_xlim(-100.0, 100.0)
        axes[1, column].set_ylim(-100.0, 100.0)
        axes[2, column].imshow(radar_panels[column], cmap="gray", vmin=0.0, vmax=1.0)
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for row, label in enumerate(["camera", "LiDAR (bird's eye)", "radar (polar)"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.tight_layout()

    out = Path(cfg["paths.figures"]) / "degradation_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print()
    print(f"montage written to {out}")

    print()
    print("note: fog veils the scene with atmospheric light, so the camera mean")
    print("      rises at low severity before the exposure loss pulls it back down.")
    print("      Contrast alone stops being usable past s=3, where the read noise")
    print("      raises it again although the frame is visibly worse; the deviation")
    print("      from the clean frame is the quantity that tracks severity.")

    failures = 0
    if not np.all(np.diff(camera_deviations) > 0.0):
        print(f"FAIL: camera deviation from the clean frame is not increasing: "
              f"{[round(v, 4) for v in camera_deviations]}")
        failures += 1
    if not np.all(np.diff(lidar_intensities) < 0.0):
        print(f"FAIL: LiDAR intensity is not monotonically decreasing: "
              f"{[round(v, 3) for v in lidar_intensities]}")
        failures += 1
    if not np.all(np.diff(radar_floors) > 0.0):
        print(f"FAIL: radar noise floor is not monotonically increasing: "
              f"{[round(v, 4) for v in radar_floors]}")
        failures += 1
    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("OK: camera deviation from the clean frame, LiDAR intensity and")
    print("    radar noise floor all move monotonically with severity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


