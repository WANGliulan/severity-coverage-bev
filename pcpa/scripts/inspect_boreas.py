"""Inspect a downloaded Boreas sequence without needing the official devkit.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\inspect_boreas.py ^
        data\\raw\\boreas\\boreas-2020-11-26-13-58

It prints what was downloaded, parses the calibrations, the DMI (wheel
encoder) trace and every LiDAR frame, and reports a summary. It is a
sanity check that the credential-free download produced usable data.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

# pyboreas stores each LiDAR point as six float32 values: x, y, z,
# intensity, range, time.
LIDAR_DTYPE = np.dtype(np.float32)
LIDAR_COLS = 6
COLS = ["x", "y", "z", "intensity", "range", "time"]


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:,.1f} PB"


def report_tree(root: str) -> None:
    print(f"sequence root: {root}")
    total = 0
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            continue
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
        total += size
        print(f"  {folder:<10} {len(files):>5} files   {human(size)}")
    print(f"  {'total':<10} {'':>5}         {human(total)}")
    print()


def report_calib(root: str) -> None:
    calib_dir = os.path.join(root, "calib")
    if not os.path.isdir(calib_dir):
        print("calib/: not downloaded")
        return
    print("calib/")
    for name in sorted(os.listdir(calib_dir)):
        path = os.path.join(calib_dir, name)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = [ln.rstrip() for ln in handle if ln.strip()]
        print(f"  {name:<26} {len(lines)} non-empty line(s)")
        if name.startswith("T_") and lines:
            try:
                mat = np.array([[float(v) for v in ln.replace(",", " ").split()]
                                for ln in lines if ln.strip()])
                if mat.shape == (4, 4):
                    print(f"      -> 4x4 transform, translation = "
                          f"{np.round(mat[:3, 3], 3).tolist()}")
            except ValueError:
                pass
    print()


def report_applanix(root: str) -> None:
    app_dir = os.path.join(root, "applanix")
    if not os.path.isdir(app_dir):
        print("applanix/: not downloaded")
        return
    print("applanix/ (vehicle state --- drives the contact-channel model)")
    dmi = os.path.join(app_dir, "dmi.csv")
    if os.path.exists(dmi):
        # the trailing line of these CSVs is sometimes truncated, so
        # tolerate malformed rows instead of aborting the whole report
        try:
            data = np.genfromtxt(dmi, delimiter=",", names=True, invalid_raise=False)
            t = np.atleast_1d(np.asarray(data["GPSTime"], dtype=float))
            pulses = np.atleast_1d(np.asarray(data["pulse_count"], dtype=float))
            good = np.isfinite(t) & np.isfinite(pulses)
            t, pulses = t[good], pulses[good]
            dt = np.diff(t)
            usable = dt > 0
            rate = (np.diff(pulses)[usable] / dt[usable]).mean() if usable.any() else float("nan")
            span = t[-1] - t[0] if len(t) else 0.0
            print(f"  dmi.csv            {len(t):,} samples, {span:.1f} s, "
                  f"mean wheel-pulse rate {rate:,.0f} counts/s")
        except Exception as exc:  # noqa: BLE001
            print(f"  dmi.csv            could not parse ({exc})")
    for name in sorted(os.listdir(app_dir)):
        if name == "dmi.csv":
            continue
        size = os.path.getsize(os.path.join(app_dir, name))
        print(f"  {name:<18} {human(size)}")
    print()


def report_lidar(root: str) -> None:
    pattern = os.path.join(root, "lidar", "*.bin")
    files = sorted(glob.glob(pattern))
    print(f"lidar/ ({len(files)} frame(s))")
    for path in files:
        points = np.fromfile(path, dtype=LIDAR_DTYPE)
        if points.size % LIDAR_COLS:
            print(f"  {os.path.basename(path)}: size is not a multiple of "
                  f"{LIDAR_COLS} floats --- unexpected layout")
            continue
        points = points.reshape(-1, LIDAR_COLS)
        xyz = points[:, :3]
        keep = np.linalg.norm(xyz, axis=1) > 0.1
        span = xyz[keep].max(axis=0) - xyz[keep].min(axis=0) if keep.any() else xyz.max(axis=0) - xyz.min(axis=0)
        print(f"  {os.path.basename(path)}: {points.shape[0]:,} points, "
              f"range {np.linalg.norm(xyz, axis=1).max():.1f} m, "
              f"extent {np.round(span, 1).tolist()} m, "
              f"intensity {points[:, 3].min():.2f}..{points[:, 3].max():.2f}")
    print()


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "boreas", "boreas-2020-11-26-13-58")
    if not os.path.isdir(root):
        print(f"not found: {root}")
        return 1
    report_tree(root)
    report_calib(root)
    report_applanix(root)
    report_lidar(root)
    print("OK --- data is readable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
