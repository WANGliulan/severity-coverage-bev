"""Object detection on the BEV grid, on top of the same fusion trunk.

    DetectionConfig      the few decisions that define the head and the decode
    detection_targets    labels -> centre heatmap, size, offset and heading
    decode_detections    head output -> scored boxes for the AP metric
    detection_loss       the training objective for the head
    Detector             wraps a trunk, so concat and gate are compared exactly
                         as they are elsewhere in the paper

The head is centre-based: one heatmap per class, and at each object's centre cell
a size, a sub-cell offset and a heading. That keeps the whole detection stage a
function of the same BEV representation the occupancy study uses, so a difference
between two fusion structures is a difference in the trunk and not in the task
definition.

The wrapper is deliberate. The two trunks keep their existing forward() semantics
--- every script in the repository depends on them --- and only gain a
``forward_features`` method, which returns the decoder features the head needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..eval.detection import Box, mean_average_precision
from .bev import BevGrid
from .network import ConvBlock


@dataclass(frozen=True)
class DetectionConfig:
    """Everything the head and the decode need to agree on."""

    classes: tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")
    hidden_channels: int = 32
    min_score: float = 0.05
    max_detections: int = 40
    peak_window: int = 3
    #: A centre cell keeps a regression target only within this radius, in cells.
    target_radius: int = 1


class DetectionHead(nn.Module):
    """Per-cell class scores and box parameters at the trunk's finest resolution."""

    def __init__(self, in_channels: int, config: DetectionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectionConfig()
        self.stem = ConvBlock(in_channels, self.config.hidden_channels)
        self.heatmap = nn.Conv2d(self.config.hidden_channels,
                                 len(self.config.classes), 1)
        self.size = nn.Conv2d(self.config.hidden_channels, 2, 1)
        self.offset = nn.Conv2d(self.config.hidden_channels, 2, 1)
        self.heading = nn.Conv2d(self.config.hidden_channels, 2, 1)
        # A small negative bias on the heatmap keeps the first batches from
        # saturating the focal term, which is the usual trick for centre heads.
        nn.init.constant_(self.heatmap.bias, -2.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.stem(features)
        return {"heatmap": self.heatmap(hidden),
                "size": self.size(hidden),
                "offset": self.offset(hidden),
                "heading": self.heading(hidden)}


class Detector(nn.Module):
    """A fusion trunk plus a detection head, trained on the occupancy loss too.

    The occupancy branch is kept: the paper's protocol evaluates both tasks, and a
    trunk trained only for boxes would not be the same object as the one the
    occupancy tables describe.
    """

    def __init__(self, trunk: nn.Module, in_channels: int,
                 config: DetectionConfig | None = None) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = DetectionHead(in_channels, config)

    @property
    def config(self) -> DetectionConfig:
        return self.head.config

    def forward(self, x: torch.Tensor,
                severity_override: int | None = None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """Return ``(occupancy logits, detection output, auxiliary logits)``."""
        features, auxiliary = self.trunk.fuse(x, severity_override)
        return self.trunk.occupancy_from_features(features), self.head(features), auxiliary


@dataclass(frozen=True)
class DetectionTargets:
    """What the head is asked to produce, per frame."""

    heatmap: np.ndarray     # (C, H, W), Gaussian peaks at object centres
    parameters: np.ndarray  # (6, H, W): length, width, dx, dy, sin, cos
    mask: np.ndarray        # (1, H, W), 1 where the parameters are supervised


def cell_center(index: int, grid: BevGrid) -> float:
    """Coordinate of the centre of cell ``index`` along its axis, in metres."""
    return (index + 0.5) * grid.resolution_m - grid.extent_m


def _gaussian(plane: np.ndarray, row: int, column: int, radius: int) -> None:
    """Stamp a Gaussian peak into ``plane``, keeping the larger value.

    Taking the maximum rather than the sum is what makes two objects in adjacent
    cells produce two peaks instead of one blurred one.
    """
    sigma = max(radius, 1) / 3.0
    offsets = np.arange(-radius, radius + 1)
    columns, rows = np.meshgrid(offsets, offsets)
    peak = np.exp(-(columns ** 2 + rows ** 2) / (2.0 * sigma * sigma))
    row_slice = slice(max(row - radius, 0), min(row + radius + 1, plane.shape[0]))
    column_slice = slice(max(column - radius, 0), min(column + radius + 1, plane.shape[1]))
    peak = peak[row_slice.start - (row - radius):peak.shape[0] - ((row + radius + 1) - row_slice.stop),
                column_slice.start - (column - radius):peak.shape[1] - ((column + radius + 1) - column_slice.stop)]
    np.maximum(plane[row_slice, column_slice], peak, out=plane[row_slice, column_slice])


def detection_targets(boxes: list[Box], grid: BevGrid,
                      config: DetectionConfig | None = None) -> DetectionTargets:
    """Encode labelled boxes as a centre heatmap plus per-cell parameters.

    Boxes whose centre falls outside the BEV grid are dropped: the grid spans
    +/-50 m while the labels cover +/-75 m, so the reach of this experiment is
    stated in the paper rather than silently extended.
    """
    config = config or DetectionConfig()
    classes = list(config.classes)
    heatmap = np.zeros((len(classes), *grid.shape), dtype=np.float32)
    parameters = np.zeros((6, *grid.shape), dtype=np.float32)
    mask = np.zeros((1, *grid.shape), dtype=np.float32)
    for box in boxes:
        if box.label not in classes:
            continue
        row, column, inside = grid.cell_of(np.array([[box.x, box.y, 0.0]]))
        if not bool(inside[0]):
            continue
        center_row, center_column = int(row[0]), int(column[0])
        radius = int(np.clip(round(max(box.length, box.width) / grid.resolution_m),
                             1, 3))
        _gaussian(heatmap[classes.index(box.label)], center_row, center_column,
                  radius)
        for delta_row in range(-config.target_radius, config.target_radius + 1):
            for delta_column in range(-config.target_radius, config.target_radius + 1):
                row_index = center_row + delta_row
                column_index = center_column + delta_column
                if not (0 <= row_index < grid.cells and 0 <= column_index < grid.cells):
                    continue
                parameters[:, row_index, column_index] = (
                    box.length, box.width,
                    box.x - cell_center(column_index, grid),
                    box.y - cell_center(row_index, grid),
                    math.sin(box.heading), math.cos(box.heading))
                mask[0, row_index, column_index] = 1.0
    return DetectionTargets(heatmap, parameters, mask)


def detection_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                   regression_weight: float = 0.5) -> torch.Tensor:
    """Centre-head loss: focal terms on the heatmap plus L1 on the parameters.

    This is the standard form from the centre-based detectors: the positive term
    is weighted by ``(1 - p)^2`` and the negative term by ``(1 - y)^4``, so the
    many empty cells contribute in proportion to how wrong they are.
    """
    heatmap_logits = outputs["heatmap"]
    target = targets["heatmap"]
    probability = torch.sigmoid(heatmap_logits).clamp(1e-4, 1.0 - 1e-4)
    positive = (target > 0.999).float()
    negative = 1.0 - positive
    positive_loss = -((1.0 - probability) ** 2) * torch.log(probability) * positive
    negative_loss = (-((1.0 - target) ** 4) * (probability ** 2)
                     * torch.log(1.0 - probability) * negative)
    count = positive.sum().clamp(min=1.0)
    heatmap_loss = (positive_loss.sum() + negative_loss.sum()) / count

    mask = targets["mask"]
    normalise = {"size": 10.0, "offset": 1.0, "heading": 1.0}
    regression = torch.zeros((), device=heatmap_logits.device)
    for name, scale in normalise.items():
        residual = torch.abs(outputs[name] - targets[name]) * mask / scale
        regression = regression + residual.sum() / mask.sum().clamp(min=1.0)
    return heatmap_loss + regression_weight * regression


@torch.no_grad()
def decode_detections(outputs: dict[str, torch.Tensor], grid: BevGrid,
                      config: DetectionConfig | None = None) -> list[list[Box]]:
    """Turn head outputs into scored boxes, one list per frame of the batch.

    A cell is kept when it is the maximum of its own neighbourhood, which is the
    suppression step: two cells of one Gaussian cannot both become detections, so
    no separate non-maximum suppression is needed.
    """
    config = config or DetectionConfig()
    probability = torch.sigmoid(outputs["heatmap"])
    pooled = nn.functional.max_pool2d(probability, config.peak_window, stride=1,
                                      padding=config.peak_window // 2)
    peaks = torch.where(pooled == probability, probability,
                        torch.zeros_like(probability))
    size, offset, heading = outputs["size"], outputs["offset"], outputs["heading"]
    frames: list[list[Box]] = []
    for index in range(probability.shape[0]):
        boxes: list[Box] = []
        for class_index, name in enumerate(config.classes):
            scores = peaks[index, class_index].reshape(-1)
            keep = min(config.max_detections, scores.numel())
            values, flat_indices = torch.topk(scores, keep)
            for value, flat in zip(values.tolist(), flat_indices.tolist()):
                if value < config.min_score:
                    break
                row, column = divmod(int(flat), probability.shape[3])
                boxes.append(Box(
                    x=cell_center(column, grid) + float(offset[index, 0, row, column]),
                    y=cell_center(row, grid) + float(offset[index, 1, row, column]),
                    length=float(size[index, 0, row, column]),
                    width=float(size[index, 1, row, column]),
                    heading=math.atan2(float(heading[index, 0, row, column]),
                                       float(heading[index, 1, row, column])),
                    score=float(value),
                    label=name))
        boxes.sort(key=lambda box: -box.score)
        frames.append(boxes[:config.max_detections])
    return frames


def _invert(targets: list[DetectionTargets]) -> dict[str, torch.Tensor]:
    """The head output that a perfect detector would produce for these targets.

    Encoding labels and then decoding them back only proves anything if the two
    are inverses, so this builds the logits that put the heatmap at its peak
    exactly on every labelled centre and copies the parameters through. Decoding
    the result must reproduce the original boxes.
    """

    def stack(part: slice) -> torch.Tensor:
        return torch.stack([torch.as_tensor(item.parameters[part])
                            for item in targets])

    heatmap = torch.stack([torch.as_tensor(item.heatmap) for item in targets])
    heatmap_logits = torch.where(heatmap > 0.999, torch.full_like(heatmap, 10.0),
                                 torch.full_like(heatmap, -10.0))
    return {"heatmap": heatmap_logits,
            "size": stack(slice(0, 2)),
            "offset": stack(slice(2, 4)),
            "heading": stack(slice(4, 6))}


def selftest() -> int:
    """Check the whole detection stage against a case with a known answer.

    A head that returns the encoded targets is by construction a perfect detector,
    so the AP it earns must be 1.0. Anything less means the encoding, the peak
    suppression or the metric loses information the others expect to be there ---
    which is exactly the failure that a plausible-looking number would hide.
    """
    grid = BevGrid()
    config = DetectionConfig()
    labelled = [
        [Box(-10.0, 3.0, 4.2, 1.8, 0.1, label="Car"),
         Box(6.0, -4.0, 4.6, 1.9, -1.2, label="Car")],
        [Box(0.5, 8.0, 0.7, 0.6, 0.0, label="Pedestrian")],
        [Box(20.0, 20.0, 1.8, 0.6, 2.0, label="Cyclist")],
        [],
    ]
    targets = [detection_targets(frame, grid, config) for frame in labelled]
    decoded = decode_detections(_invert(targets), grid, config)

    problems: list[str] = []
    expected = sum(len(frame) for frame in labelled)
    found = sum(len(frame) for frame in decoded)
    if found != expected:
        problems.append(f"decoded {found} boxes, expected {expected}")

    average = mean_average_precision(decoded, labelled)
    if average is None or average < 0.999:
        problems.append(f"perfect head: mAP {average}, expected 1.0")

    # A label outside the grid is dropped rather than silently squeezed into it.
    outside = detection_targets([Box(80.0, 0.0, 4.0, 2.0, 0.0, label="Car")],
                                grid, config)
    if outside.heatmap.max() != 0.0:
        problems.append("a box outside the grid produced a target")

    for problem in problems:
        print("FAIL: " + problem)
    print(f"{len(problems)} problem(s) in 3 cases")
    return len(problems)


if __name__ == "__main__":
    raise SystemExit(min(selftest(), 120))

