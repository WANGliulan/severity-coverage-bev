"""Dataset access: Boreas sequences, calibration, poses and labels.

Layout of a sequence, verified against the public bucket and the official
data reference::

    <sequence>/applanix/{camera,lidar,radar}_poses.csv
    <sequence>/applanix/{dmi,camera_times,lidar_times,radar_times}.*
    <sequence>/applanix/gps_post_process.csv
    <sequence>/calib/{P_camera,T_camera_lidar,T_radar_lidar,T_applanix_lidar}.txt
    <sequence>/calib/camera0_intrinsics.yaml
    <sequence>/camera/<timestamp_us>.png
    <sequence>/lidar/<timestamp_us>.bin
    <sequence>/radar/<timestamp_us>.png

The labelled subset is distributed separately under ``boreas-objects-v1``::

    boreas-objects-v1/labels_detection/<lidar_timestamp_us>.txt
    boreas-objects-v1/labels_tracking/...
    boreas-objects-v1/test_times.txt

A detection label file is whitespace separated with the columns

    uuid  type  height  width  length  x  y  z  rot_y  numPoints

where the box is expressed in the LiDAR frame, ``rot_y`` is the rotation about
the z axis, and objects are labelled within +/-75 m of the sensor. Categories
are ``Car``, ``Pedestrian``, ``Cyclist`` and ``Misc``.

One detail that bites: the per-sensor pose files carry ``GPSTime`` in
*miroseconds*, whereas ``gps_post_process.csv`` carries it in *seconds*.
Everything below returns microseconds for the per-sensor streams.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

#: A downloadable sequence is named after its start time, e.g.
#: ``boreas-2021-03-23-12-43``. The strict pattern deliberately excludes the
#: label bundle (``boreas-objects-v1``) and the dedicated calibration drives
#: (``...-cam-lidar-calib``), which are not sequences.
SEQUENCE_NAME = re.compile(r"^boreas-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")

LIDAR_POINT_DTYPE = np.dtype(np.float32)
LIDAR_POINT_COLUMNS = 6                      # x, y, z, intensity, range, time
DETECTION_CLASSES = ("Car", "Pedestrian", "Cyclist", "Misc")

POSE_COLUMNS = ("easting", "northing", "altitude",
                "vel_east", "vel_north", "vel_up",
                "roll", "pitch", "heading",
                "angvel_z", "angvel_y", "angvel_x")


def season_from_timestamp_us(timestamp_us: int) -> str:
    """Northern-hemisphere season of a frame, from its timestamp.

    Derived from the date alone, so it is verifiable; it is *not* a weather
    label. Actual weather has to be filled in by inspecting the sequence
    video, which is why the manifest keeps the two apart.
    """
    month = date.fromtimestamp(int(timestamp_us) / 1e6).month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def load_matrix(path: str | Path, rows: int, columns: int) -> np.ndarray:
    """Read a whitespace-separated matrix, e.g. a 4x4 extrinsic."""
    values = np.asarray(np.loadtxt(path, dtype=float), dtype=float).reshape(-1)
    if values.size != rows * columns:
        raise ValueError(f"{path}: expected {rows * columns} values, got {values.size}")
    return values.reshape(rows, columns)


@dataclass
class Calibration:
    """Intrinsics and extrinsics of one sequence."""

    projection_camera: np.ndarray       # 3x4, rectified projection matrix
    camera_from_lidar: np.ndarray       # 4x4
    radar_from_lidar: np.ndarray        # 4x4
    applanix_from_lidar: np.ndarray     # 4x4


def load_calibration(calib_dir: str | Path) -> Calibration:
    calib_dir = Path(calib_dir)
    return Calibration(
        # P_camera is stored as a 4x4 homogeneous matrix whose third row is
        # the projective row; only the first three rows are needed to project
        # a 3D point onto the rectified image.
        projection_camera=load_matrix(calib_dir / "P_camera.txt", 4, 4)[:3, :],
        camera_from_lidar=load_matrix(calib_dir / "T_camera_lidar.txt", 4, 4),
        radar_from_lidar=load_matrix(calib_dir / "T_radar_lidar.txt", 4, 4),
        applanix_from_lidar=load_matrix(calib_dir / "T_applanix_lidar.txt", 4, 4),
    )


@dataclass
class Poses:
    """Per-frame poses of one sensor stream."""

    timestamp_us: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.timestamp_us.size)

    @property
    def easting(self) -> np.ndarray:
        return self.values["easting"]

    @property
    def northing(self) -> np.ndarray:
        return self.values["northing"]

    def speed_m_per_s(self) -> np.ndarray:
        """Ground speed from the east/north velocity components."""
        return np.hypot(self.values["vel_east"], self.values["vel_north"])


def load_poses(path: str | Path) -> Poses:
    """Read a ``*_poses.csv`` file."""
    data = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    names = list(data.dtype.names or ())
    if "GPSTime" not in names:
        raise ValueError(f"{path}: no GPSTime column (found {names})")
    timestamps = np.atleast_1d(np.asarray(data["GPSTime"], dtype=float))
    good = np.isfinite(timestamps)
    values = {name: np.atleast_1d(np.asarray(data[name], dtype=float))[good]
              for name in POSE_COLUMNS if name in names}
    return Poses(timestamp_us=timestamps[good].astype(np.int64), values=values)


@dataclass(frozen=True)
class DetectionLabel:
    """One 3D bounding box in the LiDAR frame."""

    uuid: str
    category: str
    height_m: float
    width_m: float
    length_m: float
    x_m: float
    y_m: float
    z_m: float
    rotation_y_rad: float
    num_points: int

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m, self.z_m], dtype=float)

    @property
    def size(self) -> np.ndarray:
        return np.array([self.length_m, self.width_m, self.height_m], dtype=float)


def load_detection_labels(path: str | Path) -> list[DetectionLabel]:
    """Read one ``labels_detection`` text file."""
    labels: list[DetectionLabel] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                labels.append(DetectionLabel(
                    uuid=parts[0],
                    category=parts[1],
                    height_m=float(parts[2]),
                    width_m=float(parts[3]),
                    length_m=float(parts[4]),
                    x_m=float(parts[5]),
                    y_m=float(parts[6]),
                    z_m=float(parts[7]),
                    rotation_y_rad=float(parts[8]),
                    num_points=int(float(parts[9])),
                ))
            except ValueError:
                continue
    return labels


def labels_to_array(labels: list[DetectionLabel]) -> np.ndarray:
    """Stack labels as ``(N, 8)``: x, y, z, l, w, h, rot_y, class index."""
    if not labels:
        return np.zeros((0, 8), dtype=float)
    index = {name: i for i, name in enumerate(DETECTION_CLASSES)}
    return np.array([[lb.x_m, lb.y_m, lb.z_m, lb.length_m, lb.width_m, lb.height_m,
                      lb.rotation_y_rad, index.get(lb.category, len(DETECTION_CLASSES) - 1)]
                     for lb in labels], dtype=float)


def load_lidar_frame(path: str | Path) -> np.ndarray:
    """Read a ``.bin`` LiDAR frame as an ``(N, 6)`` float array."""
    points = np.fromfile(path, dtype=LIDAR_POINT_DTYPE)
    if points.size % LIDAR_POINT_COLUMNS:
        raise ValueError(f"{path}: size is not a multiple of {LIDAR_POINT_COLUMNS} floats")
    return points.reshape(-1, LIDAR_POINT_COLUMNS)


def load_camera_frame(path: str | Path) -> np.ndarray:
    """Read a camera PNG as an RGB uint8 array."""
    return np.asarray(Image.open(path).convert("RGB"))


def load_radar_frame(path: str | Path) -> np.ndarray:
    """Read a Navtech polar radar PNG."""
    return np.asarray(Image.open(path))


def frame_timestamp(path: str | Path) -> int:
    """The microsecond timestamp encoded in a frame filename."""
    return int(Path(path).stem)


class BoreasSequence:
    """Read-only access to one downloaded Boreas sequence.

    Frame timestamps are LiDAR/camera/radar specific and are *not* shared, so
    :meth:`nearest` is the way to pick the frame of one sensor that matches a
    timestamp of another (the label files are keyed by LiDAR timestamps).
    """

    FRAME_PATTERNS = {"camera": "*.png", "lidar": "*.bin", "radar": "*.png"}

    def __init__(self, root: str | Path, objects_root: str | Path | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"sequence directory not found: {self.root}")
        self.objects_root = Path(objects_root) if objects_root else None
        self._frames: dict[str, list[Path]] = {}
        self._timestamps: dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        return f"BoreasSequence({self.name})"

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def date_str(self) -> str:
        return self.name.replace("boreas-", "")[:10]

    # ---- frames -------------------------------------------------------
    def frames(self, kind: str) -> list[Path]:
        """Sorted frame paths of one sensor."""
        if kind not in self.FRAME_PATTERNS:
            raise KeyError(f"unknown sensor {kind!r}; expected one of {list(self.FRAME_PATTERNS)}")
        if kind not in self._frames:
            folder = self.root / kind
            self._frames[kind] = sorted(folder.glob(self.FRAME_PATTERNS[kind])) if folder.is_dir() else []
        return self._frames[kind]

    def timestamps(self, kind: str) -> np.ndarray:
        """Frame timestamps in microseconds."""
        if kind not in self._timestamps:
            self._timestamps[kind] = np.array([frame_timestamp(p) for p in self.frames(kind)],
                                              dtype=np.int64)
        return self._timestamps[kind]

    def nearest(self, kind: str, timestamp_us: int) -> Path | None:
        """Frame of ``kind`` whose timestamp is closest to ``timestamp_us``."""
        stamps = self.timestamps(kind)
        if stamps.size == 0:
            return None
        return self.frames(kind)[int(np.argmin(np.abs(stamps - int(timestamp_us))))]

    def load(self, kind: str, index: int) -> np.ndarray:
        """Load frame ``index`` of one sensor."""
        path = self.frames(kind)[index]
        if kind == "lidar":
            return load_lidar_frame(path)
        if kind == "camera":
            return load_camera_frame(path)
        return load_radar_frame(path)

    # ---- poses and calibration ---------------------------------------
    def poses(self, kind: str) -> Poses:
        """Per-frame poses of one sensor stream."""
        return load_poses(self.root / "applanix" / f"{kind}_poses.csv")

    def calibration(self) -> Calibration:
        return load_calibration(self.root / "calib")

    def vehicle_state(self) -> np.ndarray:
        """Wheel-encoder counts against time, as ``(N, 2)``."""
        path = self.root / "applanix" / "dmi.csv"
        data = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
        time_s = np.atleast_1d(np.asarray(data["GPSTime"], dtype=float))
        pulses = np.atleast_1d(np.asarray(data["pulse_count"], dtype=float))
        good = np.isfinite(time_s) & np.isfinite(pulses)
        return np.stack([time_s[good], pulses[good]], axis=1)

    # ---- labels -------------------------------------------------------
    @property
    def detection_label_dir(self) -> Path | None:
        """Folder holding ``<lidar_timestamp_us>.txt`` labels, if present.

        The labelled bundle (``boreas-objects-v1``) is a *flat* dataset: it
        carries its own ``camera``/``lidar``/``radar``/``calib``/``applanix``
        folders *and* ``labels_detection`` at the same level. The labels are
        therefore looked for inside the sequence root as well as in a
        separate objects root.
        """
        candidates = [self.root / "labels_detection"]
        if self.objects_root is not None:
            candidates.append(self.objects_root / "labels_detection")
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def label_timestamps(self, *, require_frames: bool = True) -> np.ndarray:
        """LiDAR timestamps that carry detection labels.

        With ``require_frames`` true (the default) only labels whose LiDAR
        frame is actually present locally are returned. The label bundle is
        distributed separately from the sequences and does *not* cover them
        all --- it carries its own recordings --- so without this filter a
        bundle sitting next to unlabelled sequences would make them look
        labelled.
        """
        folder = self.detection_label_dir
        if folder is None:
            return np.array([], dtype=np.int64)
        stamps = np.array(sorted(int(p.stem) for p in folder.glob("*.txt")), dtype=np.int64)
        if not require_frames:
            return stamps
        own = self.timestamps("lidar")
        if own.size == 0:
            return stamps
        return stamps[np.isin(stamps, own)]

    def labels_for(self, lidar_timestamp_us: int) -> list[DetectionLabel]:
        folder = self.detection_label_dir
        if folder is None:
            return []
        path = folder / f"{int(lidar_timestamp_us)}.txt"
        return load_detection_labels(path) if path.is_file() else []

    def test_timestamps(self) -> np.ndarray:
        """Official held-out frame timestamps, when the bundle provides them."""
        candidates = [self.root / "test_times.txt"]
        if self.objects_root is not None:
            candidates.append(self.objects_root / "test_times.txt")
        for candidate in candidates:
            if candidate.is_file():
                return np.atleast_1d(np.loadtxt(candidate, dtype=np.int64))
        return np.array([], dtype=np.int64)

    def summary(self) -> dict[str, object]:
        """A one-line description, used by the CLI and the demo script."""
        lidar_stamps = self.timestamps("lidar")
        span_s = float((lidar_stamps[-1] - lidar_stamps[0]) / 1e6) if lidar_stamps.size > 1 else 0.0
        return {
            "sequence": self.name,
            "date": self.date_str,
            "season": season_from_timestamp_us(int(lidar_stamps[0])) if lidar_stamps.size else "unknown",
            "n_camera": len(self.frames("camera")),
            "n_lidar": len(self.frames("lidar")),
            "n_radar": len(self.frames("radar")),
            "duration_s": round(span_s, 1),
            "n_labelled": int(self.label_timestamps().size),
        }


MANIFEST_COLUMNS = ("sample_id", "source", "sequence", "session_id", "timestamp_us",
                    "date", "season", "weather", "severity",
                    "camera_path", "lidar_path", "radar_path",
                    "has_labels", "n_objects", "friction_mu")


def discover_sequences(data_root: str | Path, objects_root: str | Path | None = None) -> list[BoreasSequence]:
    """Every downloaded sequence under ``data_root``, sorted by name."""
    root = Path(data_root)
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and SEQUENCE_NAME.match(p.name)):
        try:
            found.append(BoreasSequence(folder, objects_root))
        except FileNotFoundError:
            continue
    return found


def build_manifest(
    sequences: list[BoreasSequence],
    out_csv: str | Path | None = None,
    limit_per_sequence: int | None = None,
    labelled_only: bool = False,
    weather_by_sequence: dict[str, str] | None = None,
    cfg=None,
):
    """One row per LiDAR frame, with the matching camera and radar frames.

    ``weather`` stays ``unlabelled`` unless a mapping is supplied: Boreas ships
    no per-frame weather annotation, and guessing one from the date would
    contaminate the cross-weather split later on. ``season`` *is* filled in,
    because it follows from the date alone.
    """
    import pandas as pd

    weather_by_sequence = weather_by_sequence or {}
    known_weather = {}
    if cfg is not None:
        known_weather = cfg.get_path("surface.friction_mu_by_weather", default={}) or {}

    rows = []
    for sequence in sequences:
        weather = weather_by_sequence.get(sequence.name, "unlabelled")
        mu = float(known_weather[weather]) if weather in known_weather else float("nan")
        labelled = set(int(t) for t in sequence.label_timestamps().tolist())
        taken = 0
        for lidar_path in sequence.frames("lidar"):
            timestamp = frame_timestamp(lidar_path)
            has_labels = timestamp in labelled
            if labelled_only and not has_labels:
                continue
            camera_path = sequence.nearest("camera", timestamp)
            radar_path = sequence.nearest("radar", timestamp)
            taken += 1
            rows.append({
                "sample_id": f"{sequence.name}-{timestamp}",
                "source": "boreas",
                "sequence": sequence.name,
                "session_id": sequence.date_str,
                "timestamp_us": timestamp,
                "date": sequence.date_str,
                "season": season_from_timestamp_us(timestamp),
                "weather": weather,
                "severity": 0,
                "camera_path": str(camera_path) if camera_path else "",
                "lidar_path": str(lidar_path),
                "radar_path": str(radar_path) if radar_path else "",
                "has_labels": has_labels,
                "n_objects": len(sequence.labels_for(timestamp)) if has_labels else 0,
                "friction_mu": mu,
            })
            if limit_per_sequence and taken >= limit_per_sequence:
                break

    frame = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI: list sequences, inspect one, or write a frame index."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect and index the Boreas recordings.")
    parser.add_argument("--data-root", default=None, help="folder holding the sequences")
    parser.add_argument("--objects-root", default=None, help="folder holding boreas-objects-v1")
    parser.add_argument("--list", action="store_true", help="list downloaded sequences")
    parser.add_argument("--inspect", metavar="SEQUENCE", help="describe one sequence")
    parser.add_argument("--manifest", metavar="OUT.csv", help="write a frame index")
    parser.add_argument("--limit", type=int, default=None, help="frames per sequence")
    parser.add_argument("--labelled-only", action="store_true", help="keep labelled frames only")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root else Path("data/raw/boreas")
    objects_root = Path(args.objects_root) if args.objects_root else data_root / "boreas-objects-v1"
    sequences = discover_sequences(data_root, objects_root)

    if not sequences:
        print(f"no sequences found under {data_root}")
        print("download one first, e.g.")
        print("  .\\scripts\\download_boreas.ps1 -Sequence boreas-2021-03-23-12-43 "
              "-Modalities 'calib,applanix,lidar,radar,camera' -MaxFrames 20")
        return 1

    if args.list or (not args.inspect and not args.manifest):
        print(f"{len(sequences)} sequence(s) under {data_root}")
        for sequence in sequences:
            info = sequence.summary()
            print(f"  {info['sequence']}  {info['date']}  {info['season']:<6}  "
                  f"lidar={info['n_lidar']:<6} cam={info['n_camera']:<6} "
                  f"radar={info['n_radar']:<6} labelled={info['n_labelled']:<6} "
                  f"{info['duration_s']}s")
        if not args.inspect and not args.manifest:
            return 0

    if args.inspect:
        matches = [s for s in sequences if s.name == args.inspect]
        if not matches:
            print(f"no such sequence: {args.inspect}")
            return 1
        sequence = matches[0]
        print(json.dumps(sequence.summary(), indent=2))
        calib = sequence.calibration()
        print(f"P_camera     {calib.projection_camera.shape}")
        print(f"T_cam_lidar  translation {np.round(calib.camera_from_lidar[:3, 3], 3).tolist()}")
        print(f"T_rad_lidar  translation {np.round(calib.radar_from_lidar[:3, 3], 3).tolist()}")

    if args.manifest:
        frame = build_manifest(
            sequences,
            out_csv=args.manifest,
            limit_per_sequence=args.limit,
            labelled_only=args.labelled_only,
        )
        print(f"wrote {len(frame)} row(s) to {args.manifest}")
        if not frame.empty:
            print(frame.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


