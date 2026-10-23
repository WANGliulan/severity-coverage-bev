"""Metrics that operationalise the robustness constraint of the paper.

Equation~(1) requires the error to grow *slowly* with degradation severity,
rather than to be small on average. This module turns that statement into
numbers:

    error_growth_rate      slope of error against severity (the EGR)
    complementarity_gain   non-redundant benefit of an extra modality
    mean_over_seeds        mean and standard deviation across repeats

The EGR is deliberately defined on the *slope*, so a stack that is worse on
average but flatter across severity can still win --- which is the whole
point of the framework being proposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GrowthFit:
    """Least-squares fit of error against severity."""

    slope: float          # the EGR
    intercept: float      # error at severity zero
    r_squared: float
    n_levels: int

    def __str__(self) -> str:
        return (f"EGR={self.slope:+.4f}/level  intercept={self.intercept:.4f}  "
                f"R2={self.r_squared:.4f}  levels={self.n_levels}")


def error_growth_rate(
    severity: np.ndarray,
    error: np.ndarray,
) -> GrowthFit:
    """Slope of ``error`` against ``severity``, by ordinary least squares.

    ``severity`` is the severity level of the protocol (0 .. 4) and ``error``
    the task error measured at that level. Both must be one-dimensional and
    of the same length.
    """
    severity = np.asarray(severity, dtype=float).ravel()
    error = np.asarray(error, dtype=float).ravel()
    if severity.size != error.size:
        raise ValueError("severity and error must have the same length")
    if severity.size < 2:
        raise ValueError("at least two severity levels are needed for a slope")
    slope, intercept = np.polyfit(severity, error, 1)
    predicted = slope * severity + intercept
    ss_res = float(np.sum((error - predicted) ** 2))
    ss_tot = float(np.sum((error - error.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return GrowthFit(float(slope), float(intercept), float(r_squared), int(severity.size))


def complementarity_gain(
    error_baseline: np.ndarray,
    error_with_extra: np.ndarray,
) -> np.ndarray:
    """Error removed by adding a channel, per severity level.

    A value of zero at every level means the extra channel is redundant,
    which is exactly what a purely correlated modality would give.
    """
    baseline = np.asarray(error_baseline, dtype=float).ravel()
    extra = np.asarray(error_with_extra, dtype=float).ravel()
    if baseline.size != extra.size:
        raise ValueError("both error vectors must have the same length")
    return baseline - extra


def relative_complementarity_gain(
    error_baseline: np.ndarray,
    error_with_extra: np.ndarray,
) -> np.ndarray:
    """As :func:`complementarity_gain`, but normalised by the baseline error."""
    baseline = np.asarray(error_baseline, dtype=float).ravel()
    gain = complementarity_gain(baseline, error_with_extra)
    return np.divide(gain, baseline, out=np.zeros_like(gain), where=np.abs(baseline) > 1e-12)


def mean_over_seeds(values: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation over repeats (columns are seeds)."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.shape[-1] > 1 else 0.0


def severity_table(
    errors: dict[str, np.ndarray],
    severity: np.ndarray | None = None,
) -> str:
    """A readable table of error curves, with one EGR per row.

    ``errors`` maps a configuration name to its error at each severity level.
    """
    names = list(errors)
    if not names:
        return "(no configurations)"
    first = np.asarray(errors[names[0]], dtype=float).ravel()
    severity = np.arange(first.size, dtype=float) if severity is None else np.asarray(severity, dtype=float)
    head = f"{'configuration':<26}" + "".join(f"{'s=' + str(int(s)):>9}" for s in severity) + f"{'EGR':>11}"
    lines = [head, "-" * len(head)]
    for name in names:
        curve = np.asarray(errors[name], dtype=float).ravel()
        fit = error_growth_rate(severity[: curve.size], curve)
        lines.append(f"{name:<26}" + "".join(f"{v:>9.4f}" for v in curve)
                     + f"{fit.slope:>+11.4f}")
    return "\n".join(lines)
