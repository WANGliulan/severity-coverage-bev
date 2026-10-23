"""Bird's-eye-view rasterisation of the three sensing modalities.

The perception task of the ablation is *map reconstruction*, not object
detection: given the degraded sensors, rebuild the occupancy of the ground
plane the vehicle needs in order to drive. That choice matters for two
reasons.

* It needs no annotation. The reference occupancy is aggregated from the
  clean LiDAR sweep of the same frame, so the task is self-supervised and
  cannot be contaminated by label noise --- and it is exactly the "map
  completion" of the paper rather than a proxy for it.
* It gives the modality ladder somewhere to live. Vision, LiDAR and radar all
  rasterise onto the same grid, so adding a modality is literally adding input
  channels, which is what the ablation varies.

Everything is expressed in the LiDAR frame: +x forward, +y left, centred on
the sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BEV_CHANNEL_NAMES = ("occupancy", "height", "intensity", "density")

#: extra "modalities" that are not sensors but stages of the framework.
#: ``contact`` is the traversability estimate of the contact channel, written
#: into the map as an attribute plane; ``memory`` is the long-horizon map
#: prior that Stage III feeds back into the front end.
CONTACT_CHANNELS = ("traversability",)
MEMORY_CHANNELS = ("memory_prior",)


@dataclass(frozen=True)
class BevGrid:
    """Geometry of the BEV raster: a regular grid over the ground plane."""

    extent_m: float = 50.0          # half-width; the grid spans [-e, +e]
    cells: int = 200                # cells per side
    z_min_m: float = -2.5
    z_max_m: float = 1.5

    @property
    def resolution_m(self) -> float:
        return 2.0 * self.extent_m / self.cells

    @property
    def shape(self) -> tuple[int, int]:
        return self.cells, self.cells

    def cell_of(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row, column and in-range mask for points in LiDAR coordinates."""
        xyz = np.asarray(xyz, dtype=float)
        mask = ((np.abs(xyz[:, 0]) <= self.extent_m)
                & (np.abs(xyz[:, 1]) <= self.extent_m)
                & (xyz[:, 2] >= self.z_min_m)
                & (xyz[:, 2] <= self.z_max_m))
        resolution = self.resolution_m
        column = np.floor((xyz[:, 0] + self.extent_m) / resolution).astype(np.int64)
        row = np.floor((xyz[:, 1] + self.extent_m) / resolution).astype(np.int64)
        inside = mask & (column >= 0) & (column < self.cells) & (row >= 0) & (row < self.cells)
        return row, column, inside


def lidar_to_bev(points: np.ndarray, grid: BevGrid | None = None) -> np.ndarray:
    """Rasterise a LiDAR frame into ``(4, H, W)`` BEV channels.

    The channels are occupancy, mean height, mean intensity and point
    density, the usual encoding for point-cloud detectors.
    """
    grid = grid or BevGrid()
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros((len(BEV_CHANNEL_NAMES), *grid.shape), dtype=np.float32)
    row, column, inside = grid.cell_of(points[:, :3])
    row, column = row[inside], column[inside]
    height = points[inside, 2]
    intensity = points[inside, 3] if points.shape[1] > 3 else np.ones(row.size)

    channels = np.zeros((len(BEV_CHANNEL_NAMES), *grid.shape), dtype=np.float32)
    counts = np.zeros(grid.shape, dtype=np.float32)
    np.add.at(counts, (row, column), 1.0)
    np.add.at(channels[1], (row, column), height)
    np.add.at(channels[2], (row, column), intensity)

    occupied = counts > 0
    channels[0] = occupied.astype(np.float32)
    channels[1][occupied] /= counts[occupied]
    channels[2][occupied] /= counts[occupied]
    channels[3] = np.log1p(counts)
    return channels


def occupancy_reference(points: np.ndarray, grid: BevGrid | None = None) -> np.ndarray:
    """Reference occupancy of the clean sweep, as ``(1, H, W)``.

    This is the target of the perception task, built from the *undegraded*
    sweep of the same frame, so no annotation is involved.
    """
    return lidar_to_bev(points, grid)[:1]


def radar_to_bev(
    frame: np.ndarray,
    grid: BevGrid | None = None,
    max_range_m: float = 200.0,
) -> np.ndarray:
    """Rasterise a Navtech polar radar frame into ``(1, H, W)`` BEV.

    The radar image has azimuth along axis 0 (one full rotation) and range
    along axis 1, so every pixel already carries its polar coordinates and can
    be scattered onto the same grid the LiDAR uses. Several polar samples land
    in one cell, so the cell keeps the strongest return.
    """
    grid = grid or BevGrid()
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[..., 0]
    if frame.size == 0:
        return np.zeros((1, *grid.shape), dtype=np.float32)
    # The radar frames are stored as 8-bit images; scale them so that the
    # radar channel lives on the same [0, 1] range as the LiDAR channels.
    if frame.max() > 1.5:
        frame = frame / 255.0
    azimuths = (np.arange(frame.shape[0], dtype=float) + 0.5) / frame.shape[0] * 2.0 * np.pi
    distances = (np.arange(frame.shape[1], dtype=float) + 0.5) / frame.shape[1] * max_range_m
    azimuth_grid, range_grid = np.meshgrid(azimuths, distances, indexing="ij")
    x = (range_grid * np.cos(azimuth_grid)).ravel()
    y = (range_grid * np.sin(azimuth_grid)).ravel()
    row, column, inside = grid.cell_of(np.column_stack([x, y, np.zeros_like(x)]))
    values = frame.ravel()
    bev = np.zeros(grid.shape, dtype=np.float32)
    np.maximum.at(bev, (row[inside], column[inside]), values[inside])
    return bev[None, ...]


def camera_context_batch(images: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """A deliberately weak fallback: global brightness and contrast planes.

    Kept only for the case where no calibration is available. It carries no
    spatial information at all, so a model fed this cannot improve with
    severity --- which is why :func:`camera_to_bev_batch` is the default.
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 4:
        images = images.mean(axis=-1)
    if images.size and images.max() > 1.5:
        images = images / 255.0
    count = images.shape[0]
    brightness = np.repeat(images.mean(axis=(1, 2))[:, None, None],
                           target_shape[0] * target_shape[1], axis=1).reshape(
                               count, *target_shape).astype(np.float32)
    contrast = np.repeat(images.std(axis=(1, 2))[:, None, None],
                         target_shape[0] * target_shape[1], axis=1).reshape(
                             count, *target_shape).astype(np.float32)
    return np.stack([brightness, contrast], axis=1)


@dataclass(frozen=True)
class CameraCalibration:
    """What inverse perspective mapping needs from a camera."""

    projection: np.ndarray          # 3x4 rectified projection
    camera_from_lidar: np.ndarray   # 4x4
    height_m: float = 1.30          # camera height above the ground plane
    image_size: tuple[int, int] = (2048, 2448)   # rows, columns of the source

    @property
    def focal_y(self) -> float:
        return float(self.projection[1, 1])

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.projection[0, 2]), float(self.projection[1, 2])

    @classmethod
    def from_calibration(cls, calibration, height_m: float = 1.30,
                         image_size: tuple[int, int] = (2048, 2448)) -> "CameraCalibration":
        return cls(projection=calibration.projection_camera,
                   camera_from_lidar=calibration.camera_from_lidar,
                   height_m=height_m,
                   image_size=image_size)

    def rescaled(self, target_size: tuple[int, int]) -> "CameraCalibration":
        """The same camera, expressed for a differently sized image.

        Needed because the intrinsic matrix is published for the native
        2448x2048 frames while the pipeline may work on downsampled ones:
        scale the focal lengths and the principal point, keep the extrinsics.
        """
        source_rows, source_columns = self.image_size
        target_rows, target_columns = target_size
        scale_x = target_columns / source_columns
        scale_y = target_rows / source_rows
        projection = np.array(self.projection, dtype=float, copy=True)
        projection[0, 0] *= scale_x
        projection[0, 2] *= scale_x
        projection[1, 1] *= scale_y
        projection[1, 2] *= scale_y
        return CameraCalibration(projection=projection,
                                 camera_from_lidar=self.camera_from_lidar,
                                 height_m=self.height_m,
                                 image_size=target_size)


def camera_to_bev_ipm(
    image: np.ndarray,
    calib: CameraCalibration,
    grid: BevGrid | None = None,
    max_range_m: float = 40.0,
    min_range_m: float = 1.0,
) -> np.ndarray:
    """Inverse perspective mapping of a single camera onto the ground plane.

    For a rectified camera at height ``h``, a pixel ``(u, v)`` below the
    horizon sees the ground at

        Z = f_y * h / (v - c_y),      X = (u - c_x) * Z / f_x

    in the camera frame (x right, y down, z forward). Those points are rotated
    into the LiDAR frame with the inverse of ``T_camera_lidar`` and splatted
    onto the BEV grid, which is the standard flat-ground inverse perspective
    mapping.

    The raw grey level is written into the channel rather than a contrast
    normalised one. That is deliberate: under fog the atmospheric light veils
    the whole image towards a constant, so the mapped cells become
    indistinguishable and the channel genuinely loses information as severity
    rises. Normalising per image would hide exactly the effect the ablation
    is meant to measure.
    """
    grid = grid or BevGrid()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    if image.size and image.max() > 1.5:
        image = image / 255.0

    height_px, width_px = image.shape
    focal_x = float(calib.projection[0, 0])
    principal_x, principal_y = calib.principal_point
    rows = np.arange(height_px, dtype=float)
    columns = np.arange(width_px, dtype=float)

    depth = np.full(height_px, np.inf, dtype=float)
    below_horizon = rows > principal_y + 1e-6
    depth[below_horizon] = (calib.focal_y * calib.height_m
                            / (rows[below_horizon] - principal_y))
    if focal_x <= 0.0:
        raise ValueError("camera projection has a non-positive focal length")

    forward = np.broadcast_to(depth[:, None], (height_px, width_px))
    lateral = (columns[None, :] - principal_x) / focal_x * forward
    visible = (np.isfinite(forward) & (forward >= min_range_m) & (forward <= max_range_m))

    points_camera = np.stack([lateral[visible],
                              np.full(visible.sum(), calib.height_m),
                              forward[visible]], axis=1)
    homogeneous = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))],
                                 axis=1)
    lidar_from_camera = np.linalg.inv(calib.camera_from_lidar)
    points_lidar = (lidar_from_camera @ homogeneous.T).T[:, :3]

    row_index, column_index, inside = grid.cell_of(points_lidar)
    values = image[visible]
    bev = np.zeros(grid.shape, dtype=np.float32)
    seen = np.zeros(grid.shape, dtype=np.float32)
    np.maximum.at(bev, (row_index[inside], column_index[inside]), values[inside])
    np.add.at(seen, (row_index[inside], column_index[inside]), 1.0)
    return np.stack([np.clip(bev, 0.0, 1.0), (seen > 0).astype(np.float32)], axis=0)


def camera_to_bev_batch(
    images: np.ndarray,
    calib: CameraCalibration | None,
    grid: BevGrid | None = None,
    max_range_m: float = 40.0,
) -> np.ndarray:
    """Inverse perspective mapping for a batch of images.

    Falls back to the (spatially uninformative) global context encoder when no
    calibration is available, and says so through the channel count, which is
    the same in both cases.
    """
    grid = grid or BevGrid()
    images = np.asarray(images)
    if calib is None:
        return camera_context_batch(images, grid.shape)
    return np.stack([camera_to_bev_ipm(image, calib, grid, max_range_m) for image in images])


def assemble_inputs(
    batch: dict[str, np.ndarray],
    modalities: tuple[str, ...],
    grid: BevGrid | None = None,
) -> np.ndarray:
    """Stack the BEV channels of the requested modalities into one tensor.

    ``batch`` maps a modality name to its data (a point cloud for ``lidar``, a
    polar image for ``radar``, image batch for ``camera``). The returned array
    is the network input for exactly those modalities, which is how the
    ablation varies the input without touching the architecture.
    """
    grid = grid or BevGrid()
    planes = []
    for modality in modalities:
        if modality == "lidar":
            planes.append(lidar_to_bev(batch["lidar"], grid))
        elif modality == "radar":
            planes.append(radar_to_bev(batch["radar"], grid))
        elif modality == "camera":
            planes.append(camera_to_bev_batch(batch["camera"],
                                              batch.get("camera_calibration"),
                                              grid))
        elif modality == "contact":
            planes.append(contact_plane(batch["mu_hat"], grid))
        elif modality == "memory":
            planes.append(np.asarray(batch["memory"], dtype=np.float32)[None, ...])
        else:
            raise KeyError(f"unknown modality {modality!r}")
    return np.concatenate(planes, axis=0)


def contact_plane(mu_hat: float, grid: BevGrid | None = None) -> np.ndarray:
    """The contact channel's contribution, as one attribute plane.

    The channel estimates the friction of the surface the vehicle is on, which
    is a property of the *place*, not of a pixel, so it enters the map as a
    constant traversability plane. Comparing a model that receives it against
    one that does not is exactly the ``+contact`` step of the ablation.
    """
    grid = grid or BevGrid()
    return np.full((1, *grid.shape), float(mu_hat), dtype=np.float32)
