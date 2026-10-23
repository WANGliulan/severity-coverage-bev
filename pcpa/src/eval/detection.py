"""Detection metrics on the bird's-eye-view plane.

    box_iou               rotated-rectangle IoU, exact, by polygon clipping
    average_precision     per-class AP over a labelled frame set
    mean_average_precision

The occupancy study needs no annotation, which is why it could be run first. This
module exists for the task that does: object detection on the labelled subset,
where the reference is a 3D box from the dataset rather than the LiDAR sweep.

AP is defined the way detection benchmarks define it. Predictions from *every*
frame are pooled, sorted by score, matched greedily to still-unmatched ground
truth at an IoU threshold, and the precision-recall curve is integrated with all
points interpolated. Averaging a per-frame metric instead would produce a
different, more flattering number that no benchmark reports, so it is not what
this module computes.

Boxes are given in the LiDAR frame with ``x`` forward, ``y`` left, ``length``
along the heading and ``width`` across it, which is the convention of the Boreas
label files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The classes the labels use. ``Misc`` is reported but left out of the mean,
#: because it is a residual category rather than an object type.
DETECTION_CLASSES = ("Car", "Pedestrian", "Cyclist", "Misc")
SCORED_CLASSES = ("Car", "Pedestrian", "Cyclist")


@dataclass(frozen=True)
class Box:
    """A 3D box reduced to its BEV footprint, with a score and a class."""

    x: float
    y: float
    length: float
    width: float
    heading: float
    score: float = 1.0
    label: str = ""

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def area(self) -> float:
        return float(self.length * self.width)

    def corners(self) -> np.ndarray:
        """The four corners in order, as ``(4, 2)``."""
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        forward = np.array([cosine, sine], dtype=float) * (self.length / 2.0)
        left = np.array([-sine, cosine], dtype=float) * (self.width / 2.0)
        offsets = np.stack([forward + left, forward - left,
                            -forward - left, -forward + left])
        return offsets + self.center


def _clip(polygon: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Sutherland-Hodgman clip of ``polygon`` to ``normal . p <= offset``."""
    if len(polygon) == 0:
        return polygon
    distances = polygon @ normal - offset
    kept: list[np.ndarray] = []
    for index in range(len(polygon)):
        following = (index + 1) % len(polygon)
        current_in = distances[index] <= 0.0
        following_in = distances[following] <= 0.0
        if current_in:
            kept.append(polygon[index])
        if current_in != following_in:
            span = distances[index] - distances[following]
            if abs(span) > 1e-12:
                kept.append(polygon[index]
                            + (polygon[following] - polygon[index])
                            * (distances[index] / span))
    return np.asarray(kept, dtype=float) if kept else np.zeros((0, 2))


def _polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area, always non-negative."""
    if len(polygon) < 3:
        return 0.0
    following = np.roll(polygon, -1, axis=0)
    return float(abs(np.sum(polygon[:, 0] * following[:, 1]
                            - following[:, 0] * polygon[:, 1])) / 2.0)


def intersection_area(first: Box, second: Box) -> float:
    """Area shared by two rotated rectangles, by clipping one with the other."""
    polygon = first.corners()
    corners = second.corners()
    for index in range(4):
        edge_start, edge_end = corners[index], corners[(index + 1) % 4]
        edge = edge_end - edge_start
        normal = np.array([-edge[1], edge[0]], dtype=float)
        if np.linalg.norm(normal) < 1e-12:
            continue
        polygon = _clip(polygon, normal, float(normal @ edge_start))
        if len(polygon) == 0:
            return 0.0
    return _polygon_area(polygon)


def box_iou(first: Box, second: Box) -> float:
    """BEV IoU of two boxes; identical boxes give 1.0, disjoint boxes 0.0."""
    intersection = intersection_area(first, second)
    union = first.area() + second.area() - intersection
    return intersection / union if union > 0.0 else 0.0


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP: the integral of the precision envelope."""
    if recall.size == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.concatenate([[recall[0]], np.diff(recall)])
    return float(np.sum(steps * envelope))


def average_precision(
    predictions: list[list[Box]],
    ground_truth: list[list[Box]],
    classes: tuple[str, ...] = DETECTION_CLASSES,
    iou_threshold: float = 0.5,
) -> dict[str, float | None]:
    """Per-class AP over a whole split.

    ``predictions`` and ``ground_truth`` are parallel lists, one entry per frame.
    A class with no ground truth anywhere returns ``None`` rather than ``0.0``, so
    that it is left out of a mean instead of dragging that mean down.
    """
    result: dict[str, float | None] = {}
    for name in classes:
        pooled: list[tuple[float, int, Box]] = []
        truth_per_frame: list[list[Box]] = []
        for frame, (frame_boxes, frame_truth) in enumerate(zip(predictions,
                                                               ground_truth)):
            truth = [box for box in frame_truth if box.label == name]
            truth_per_frame.append(truth)
            for box in frame_boxes:
                if box.label == name:
                    pooled.append((float(box.score), frame, box))
        total_truth = sum(len(truth) for truth in truth_per_frame)
        if total_truth == 0:
            result[name] = None
            continue
        pooled.sort(key=lambda item: -item[0])
        claimed = [np.zeros(len(truth), dtype=bool) for truth in truth_per_frame]
        true_positive = np.zeros(len(pooled), dtype=float)
        false_positive = np.zeros(len(pooled), dtype=float)
        for rank, (_, frame, box) in enumerate(pooled):
            truth = truth_per_frame[frame]
            best, best_index = iou_threshold, -1
            for index, candidate in enumerate(truth):
                if claimed[frame][index]:
                    continue
                overlap = box_iou(box, candidate)
                if overlap >= best:
                    best, best_index = overlap, index
            if best_index >= 0:
                claimed[frame][best_index] = True
                true_positive[rank] = 1.0
            else:
                false_positive[rank] = 1.0
        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recall = cumulative_tp / total_truth
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        result[name] = _interpolated_ap(recall, precision)
    return result


def mean_average_precision(
    predictions: list[list[Box]],
    ground_truth: list[list[Box]],
    classes: tuple[str, ...] = SCORED_CLASSES,
    iou_threshold: float = 0.5,
) -> float | None:
    """Mean of the per-class APs that have ground truth; ``None`` if none do."""
    per_class = average_precision(predictions, ground_truth, classes, iou_threshold)
    values = [value for value in per_class.values() if value is not None]
    return float(np.mean(values)) if values else None


def selftest() -> int:
    """Check the geometry and the metric against cases whose answers are known.

    An AP is easy to get subtly wrong in a way that still produces a plausible
    number, so the cases below are the ones this implementation can actually
    fail: an exact box, an overlap of exactly one third, rotation, a duplicate
    that must not count twice, and a confident false positive that must cost half
    the AP.
    """
    reference = Box(0.0, 0.0, 4.0, 2.0, 0.0, label="Car")
    problems: list[str] = []

    exact = box_iou(reference, Box(0.0, 0.0, 4.0, 2.0, 0.0))
    if abs(exact - 1.0) > 1e-9:
        problems.append(f"identical boxes: IoU {exact:.6f}, expected 1.0")

    # Intersection 2x2 = 4, union 8 + 8 - 4 = 12, so exactly one third.
    shifted = box_iou(reference, Box(2.0, 0.0, 4.0, 2.0, 0.0))
    if abs(shifted - 1.0 / 3.0) > 1e-9:
        problems.append(f"half-overlap: IoU {shifted:.6f}, expected 0.333333")

    # The same rectangle turned a quarter: intersection is the 2x2 square again.
    turned = box_iou(reference, Box(0.0, 0.0, 4.0, 2.0, math.pi / 2.0))
    if abs(turned - 1.0 / 3.0) > 1e-9:
        problems.append(f"quarter-turn: IoU {turned:.6f}, expected 0.333333")

    square = box_iou(Box(0.0, 0.0, 4.0, 4.0, 0.0),
                     Box(0.0, 0.0, 4.0, 4.0, math.pi / 2.0))
    if abs(square - 1.0) > 1e-9:
        problems.append(f"turned square: IoU {square:.6f}, expected 1.0")

    disjoint = box_iou(reference, Box(20.0, 0.0, 4.0, 2.0, 0.0))
    if disjoint != 0.0:
        problems.append(f"disjoint boxes: IoU {disjoint:.6f}, expected 0.0")

    truth = [[Box(0.0, 0.0, 4.0, 2.0, 0.0, label="Car")]]

    found = [[Box(0.0, 0.0, 4.0, 2.0, 0.0, score=0.9, label="Car")]]
    if abs(mean_average_precision(found, truth, ("Car",)) - 1.0) > 1e-9:
        problems.append("a single correct detection did not give AP 1.0")

    duplicate = [[Box(0.0, 0.0, 4.0, 2.0, 0.0, score=0.9, label="Car"),
                  Box(0.0, 0.0, 4.0, 2.0, 0.0, score=0.8, label="Car")]]
    if abs(mean_average_precision(duplicate, truth, ("Car",)) - 1.0) > 1e-9:
        problems.append("a duplicate after recall 1.0 should leave AP at 1.0")

    confident_wrong = [[Box(20.0, 0.0, 4.0, 2.0, 0.0, score=0.9, label="Car"),
                        Box(0.0, 0.0, 4.0, 2.0, 0.0, score=0.8, label="Car")]]
    halved = mean_average_precision(confident_wrong, truth, ("Car",))
    if halved is None or abs(halved - 0.5) > 1e-9:
        problems.append(f"a confident false positive: AP {halved}, expected 0.5")

    missed = mean_average_precision([[]], truth, ("Pedestrian",))
    if missed is not None:
        problems.append(f"a class with no truth: AP {missed}, expected None")

    for problem in problems:
        print("FAIL: " + problem)
    print(f"{len(problems)} problem(s) in 9 cases")
    return len(problems)


if __name__ == "__main__":
    raise SystemExit(min(selftest(), 120))
