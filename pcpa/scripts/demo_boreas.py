"""Smoke test for the Boreas loader.

Run from the project root::

    .\\.venv\\Scripts\\python.exe scripts\\demo_boreas.py

Verifies, against the real files on disk, that the calibration parses into
orthonormal rotations, that poses and the wheel-encoder trace load, that the
labelled subset is found and parsed, and that the camera and radar frames
matched to a LiDAR timestamp really are close in time. It also writes a small
manifest so the rest of the pipeline has something to consume.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config                              # noqa: E402
from src.data.boreas import (                                          # noqa: E402
    BoreasSequence,
    build_manifest,
    discover_sequences,
    frame_timestamp,
    labels_to_array,
    load_lidar_frame,
)

CAMERA_MATCH_TOLERANCE_US = 60_000      # camera runs at 10 Hz


def main() -> int:
    cfg = load_config()
    data_root = Path(cfg["paths.data_raw"]) / "boreas"
    objects_root = data_root / "boreas-objects-v1"

    sequences = discover_sequences(data_root, objects_root)
    print(f"data root : {data_root}")
    print(f"objects   : {objects_root}  (present: {objects_root.is_dir()})")
    print(f"sequences : {len(sequences)}")
    for sequence in sequences:
        info = sequence.summary()
        print(f"  {info['sequence']}  {info['season']:<6} lidar={info['n_lidar']:<6} "
              f"cam={info['n_camera']:<6} radar={info['n_radar']:<6} "
              f"labelled={info['n_labelled']}")
    if not sequences:
        print("nothing to check: download a sequence first")
        return 1

    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        print(f"[{'OK  ' if ok else 'FAIL'}] {name:<34} {detail}")
        if not ok:
            failures += 1

    sequence = sequences[0]
    print(f"\ninspecting {sequence.name}")

    calib = sequence.calibration()
    for label, matrix in (("T_camera_lidar", calib.camera_from_lidar),
                          ("T_radar_lidar", calib.radar_from_lidar),
                          ("T_applanix_lidar", calib.applanix_from_lidar)):
        rotation = matrix[:3, :3]
        residual = float(np.abs(rotation @ rotation.T - np.eye(3)).max())
        check(f"{label} rotation orthonormal", residual < 1e-6, f"max|RR^T-I|={residual:.2e}")
    focal = float(calib.projection_camera[0, 0])
    check("camera projection sane", focal > 0.0 and calib.projection_camera.shape == (3, 4),
          f"f={focal:.1f}px  shape={calib.projection_camera.shape}")

    poses_path = sequence.root / "applanix" / "lidar_poses.csv"
    if poses_path.is_file():
        lidar_poses = sequence.poses("lidar")
        check("lidar poses load", len(lidar_poses) > 0 and "easting" in lidar_poses.values,
              f"{len(lidar_poses)} rows, median speed "
              f"{np.nanmedian(lidar_poses.speed_m_per_s()):.2f} m/s")
    else:
        print("[    ] lidar_poses.csv not downloaded for this sequence "
              "(-Modalities applanix fetches it)")

    dmi_path = sequence.root / "applanix" / "dmi.csv"
    if dmi_path.is_file():
        dmi = sequence.vehicle_state()
        check("wheel encoder loads", dmi.shape[0] > 0, f"{dmi.shape[0]} samples")
    else:
        print("[    ] dmi.csv not downloaded for this sequence")

    stamps = sequence.timestamps("lidar")
    if stamps.size:
        offsets = []
        for timestamp in stamps[:20]:
            camera_path = sequence.nearest("camera", int(timestamp))
            if camera_path is not None:
                offsets.append(abs(frame_timestamp(camera_path) - int(timestamp)))
        if offsets:
            worst = max(offsets)
            check("camera matched within tolerance", worst <= CAMERA_MATCH_TOLERANCE_US,
                  f"worst |dt| = {worst / 1000.0:.1f} ms over {len(offsets)} frames")
        else:
            check("camera matched within tolerance", False, "no camera frames present")

    labelled = sequence.label_timestamps()
    if labelled.size:
        sample_labels = sequence.labels_for(int(labelled[0]))
        array = labels_to_array(sample_labels)
        check("detection labels parse", array.shape[0] == len(sample_labels) and array.shape[1] == 8,
              f"{labelled.size} labelled frames, first frame has {array.shape[0]} objects")
        test_stamps = sequence.test_timestamps()
        check("official test split present", test_stamps.size > 0, f"{test_stamps.size} test frames")
    else:
        print("[    ] no detection labels downloaded (only the calibration sample is present)")

    print()
    if (objects_root / "labels_detection").is_dir():
        print(f"inspecting the labelled bundle {objects_root.name}")
        bundle = BoreasSequence(objects_root)
        stamps = bundle.label_timestamps()
        check("bundle labels found", stamps.size > 0, f"{stamps.size} labelled frames")
        if stamps.size:
            first = int(stamps[0])
            labels = bundle.labels_for(first)
            array = labels_to_array(labels)
            categories = sorted({lb.category for lb in labels})
            check("bundle labels parse", array.shape[0] == len(labels) > 0 and array.shape[1] == 8,
                  f"{len(labels)} objects in frame {first}, classes {categories}")
            lidar_path = bundle.root / "lidar" / f"{first}.bin"
            camera_path = bundle.nearest("camera", first)
            radar_path = bundle.nearest("radar", first)
            check("bundle carries its own sensors",
                  lidar_path.is_file() and camera_path is not None,
                  f"lidar={'yes' if lidar_path.is_file() else 'no'} "
                  f"camera={'yes' if camera_path else 'no'} "
                  f"radar={'yes' if radar_path else 'no'}")
            if lidar_path.is_file():
                points = load_lidar_frame(lidar_path)
                check("bundle lidar parses", points.shape[1] == 6,
                      f"{points.shape[0]:,} points, max range "
                      f"{np.linalg.norm(points[:, :3], axis=1).max():.1f} m")
            test_stamps = bundle.test_timestamps()
            check("official test split reads", test_stamps.size > 0,
                  f"{test_stamps.size} held-out frames")
    else:
        print(f"[    ] no labels_detection under {objects_root}")

    manifest = build_manifest(sequences, out_csv=cfg["paths.manifest"] / "boreas_manifest.csv",
                              limit_per_sequence=20)
    check("manifest written", not manifest.empty,
          f"{len(manifest)} rows -> {cfg['paths.manifest'] / 'boreas_manifest.csv'}")

    print()
    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("All Boreas loader checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
