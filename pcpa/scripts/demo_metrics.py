"""Self-test for the metric implementations.

Run from the project root::

    .\\.venv\\Scripts\\python.exe scripts\\demo_metrics.py

The checks are analytic: a constructed error curve with a known slope must
come back with that slope, a flat curve must give an EGR of zero, and a
redundant channel must give a complementarity gain of zero at every level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import (                                    # noqa: E402
    complementarity_gain,
    error_growth_rate,
    mean_over_seeds,
    severity_table,
)

SEVERITY = np.arange(5, dtype=float)


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        print(f"[{'OK  ' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            failures += 1

    flat = np.full(5, 0.30)
    fit = error_growth_rate(SEVERITY, flat)
    check("flat curve -> EGR 0", abs(fit.slope) < 1e-12, str(fit))

    linear = 0.20 + 0.05 * SEVERITY
    fit = error_growth_rate(SEVERITY, linear)
    check("known slope recovered", abs(fit.slope - 0.05) < 1e-12 and abs(fit.r_squared - 1.0) < 1e-12, str(fit))

    per_frame = 0.10 + 0.09 * SEVERITY          # excellent when clean, degrades fast
    closed_loop = 0.26 + 0.02 * SEVERITY        # worse when clean, almost flat
    egr_a = error_growth_rate(SEVERITY, per_frame).slope
    egr_b = error_growth_rate(SEVERITY, closed_loop).slope
    check("flatter stack wins on EGR", egr_b < egr_a,
          f"closed loop {egr_b:+.4f} < per-frame {egr_a:+.4f}")
    check("...even though it is worse on the clean end", closed_loop[0] > per_frame[0],
          f"s=0: {closed_loop[0]:.2f} vs {per_frame[0]:.2f}")
    check("...and it crosses over at high severity", closed_loop[-1] < per_frame[-1],
          f"s=4: {closed_loop[-1]:.2f} vs {per_frame[-1]:.2f}")

    redundant = complementarity_gain(per_frame, per_frame)
    check("redundant channel -> zero gain", np.allclose(redundant, 0.0), str(redundant))

    helped = complementarity_gain(per_frame, per_frame - 0.03 * SEVERITY)
    check("useful channel -> growing gain", np.all(np.diff(helped) > 0), str(np.round(helped, 4)))

    mean, std = mean_over_seeds(np.array([[1.0, 1.1, 0.9], [2.0, 2.2, 1.8]]))
    check("seed statistics", abs(mean - 1.5) < 1e-9 and std > 0.0, f"mean={mean:.3f} std={std:.3f}")

    print()
    print(severity_table({
        "per-frame, vision only": per_frame,
        "closed loop (ours)": closed_loop,
        "closed loop, no contact": per_frame - 0.03 * SEVERITY,
    }))
    print()
    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("All metric checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
