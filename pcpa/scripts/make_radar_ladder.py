"""Draw the radar operator at five severity levels, as an operator check.

    .\\.venv\\Scripts\\python.exe scripts\\make_radar_ladder.py

**What this figure is, and what it is not.** The reported runs pass the radar
stream *undegraded*: ``build_channels`` in ``run_fusion_study.py`` feeds
``radar_to_bev(radars[index], ...)``, the raw sweep, while the camera and the
LiDAR go through their operators. ``tab_ladder`` in ``make_result_tables.py``
leaves the radar rows out for the same reason. So this figure documents what the
radar branch of the operator *does*; it is not a set of inputs any reported
number was computed from, and any caption that shows it has to say so.

The panel is a Navtech polar sweep, azimuth on the vertical axis and range on
the horizontal one. A raw frame averages 10/255, so a straight render is a black
rectangle: the near field is clipped at its 99.5th percentile and lifted with a
gamma, which is how a polar sweep is normally drawn.

Outputs, all from the same five degraded frames:

* ``figs/paper/fig_radar_ladder.pdf`` and ``.png`` --- the montage, paper-sized;
* ``图素材/阶梯_雷达_s0.png`` ... ``s4.png`` --- one file per level;
* ``figures/radar_operator_check.png`` --- the same montage next to the runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
from PIL import Image                                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config                             # noqa: E402
from src.degrade import radar as rad                                  # noqa: E402
from src.degrade.severity import SeverityLadder                       # noqa: E402

OUT_PAPER = ROOT.parent / "figs" / "paper"
OUT_ASSETS = ROOT.parent / "图素材"
OUT_RUNS = ROOT / "figures"

#: Range bins kept for display: the near field is where weather clutter shows.
RANGE_BINS = 900

#: Severity chips, the same five colours the other figures use.
SEV_FILL = ["#e8eef4", "#fde3c8", "#f7bd8a", "#ef8f46", "#c2570d"]
SEV_TEXT = ["#2b3a49", "#4a3315", "#4a3315", "#ffffff", "#ffffff"]
RADAR_C = "#9467bd"
GREY = "#555555"

WIDTH_IN, HEIGHT_IN = 7.0, 1.92


def display(frame: np.ndarray, range_bins: int = RANGE_BINS) -> np.ndarray:
    """The near field of a polar sweep, contrast-stretched to [0, 1].

    Identical to ``radar_thumbnail`` in ``make_story_figures.py``, so the panel
    here and the radar thumbnail in the overview are the same rendering; a raw
    Navtech frame sits at the bottom of the 8-bit range and would be black.
    """
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = frame[..., 0]
    near = frame[:, :min(range_bins, frame.shape[1])].astype(np.float64) / 255.0
    top = float(np.percentile(near, 99.5)) or 1.0
    return np.clip(near / top, 0.0, 1.0) ** 0.45


def find_frame(data_raw: str) -> Path:
    """One radar frame from the middle of a drive the study actually uses.

    ``boreas-objects-v1`` is the labelled subset the detection experiment uses;
    it is skipped so the panel shows a drive from the fourteen the occupancy runs
    are drawn from.
    """
    root = Path(data_raw) / "boreas"
    frames = [path for path in sorted(root.glob("*/radar/*.png"))
              if "objects" not in path.parent.parent.name]
    if not frames:
        raise SystemExit(f"no radar frames under {root}; nothing to draw")
    return frames[len(frames) // 2]


def degrade_ladder(frame: np.ndarray, ladder: SeverityLadder, cfg) -> list[np.ndarray]:
    """The five levels of the radar operator, same rng seed for all of them."""
    rng = np.random.default_rng(0)
    panels = []
    for level in ladder.levels:
        params = ladder.at(float(level))
        out = rad.degrade(frame, rad.RadarDegradation(
            clutter_to_signal_db=params.radar_clutter_db,
            angular_smear_bins=params.radar_angular_smear_bins,
            looks=params.radar_looks,
            dynamic_range_db=float(cfg["degradation.radar.dynamic_range_db"])), rng)
        panels.append(out)
    return panels


def inches(figure, left: float, top: float, width: float, height: float):
    """An axis placed in inches from the top-left corner of the figure."""
    return figure.add_axes([left / WIDTH_IN, 1.0 - (top + height) / HEIGHT_IN,
                            width / WIDTH_IN, height / HEIGHT_IN])


def montage(ladder: SeverityLadder, panels: list[np.ndarray]) -> plt.Figure:
    """Five polar panels in a row, each with its level chip and its parameters."""
    figure = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))

    def fx(inches: float) -> float:
        """Inches from the left edge -> figure fraction."""
        return inches / WIDTH_IN

    def fy(inches: float) -> float:
        """Inches from the top edge -> figure fraction."""
        return 1.0 - inches / HEIGHT_IN

    left, width, gap = 0.34, 1.225, 0.085
    for column, (level, panel) in enumerate(zip(ladder.levels, panels)):
        params = ladder.at(float(level))
        x = left + column * (width + gap)
        chip = inches(figure, x, 0.34, width, 0.21)
        chip.set_facecolor(SEV_FILL[int(level)])
        chip.set_xticks([]); chip.set_yticks([])
        for spine in chip.spines.values():
            spine.set_visible(False)
        chip.text(0.5, 0.5, f"s={int(level)}", ha="center", va="center",
                  fontsize=8.0, weight="bold", color=SEV_TEXT[int(level)])

        axis = inches(figure, x, 0.58, width, 0.70)
        axis.imshow(display(panel), cmap="gray", vmin=0.0, vmax=1.0,
                    aspect="auto", interpolation="nearest")
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor(RADAR_C)
            spine.set_linewidth(0.6)

        figure.text(fx(x + width / 2), fy(1.35),
                    f"+{params.radar_clutter_db:g} dB clutter",
                    ha="center", va="top", fontsize=6.4, color="#333333")
        figure.text(fx(x + width / 2), fy(1.48),
                    f"{params.radar_angular_smear_bins:g} bin smear",
                    ha="center", va="top", fontsize=6.4, color="#333333")

    figure.text(fx(0.34), fy(0.14), "THE OPERATOR'S RADAR BRANCH",
                fontsize=6.4, color=GREY, weight="bold", va="top")
    figure.text(fx(WIDTH_IN - 0.05), fy(0.14),
                "one polar sweep, five severity levels",
                fontsize=6.4, color=GREY, ha="right", va="top")
    figure.text(fx(0.34), fy(1.68),
                "Reference only: the reported runs pass the radar through undegraded, "
                "so no number in this repository was computed from these frames.",
                fontsize=6.2, color="#8a4b00", va="top", style="italic")
    figure.text(fx(0.115), fy(0.93), "radar (polar, near field)",
                rotation=90, fontsize=6.6, color=RADAR_C, va="center", ha="center")
    return figure


def main() -> int:
    cfg = load_config()
    ladder = SeverityLadder.from_config(cfg)
    source = find_frame(str(cfg["paths.data_raw"]))
    frame = np.asarray(Image.open(source))
    print(f"radar frame   : {source}")
    print(f"frame shape   : {frame.shape}   (azimuth x range), "
          f"mean {frame.mean():.1f}/255")

    panels = degrade_ladder(frame, ladder, cfg)
    print()
    print(f"{'s':>2}{'clutter dB':>12}{'smear bins':>12}{'looks':>8}{'floor':>9}")
    for level, panel in zip(ladder.levels, panels):
        params = ladder.at(float(level))
        stats = rad.summary(panel)
        print(f"{int(level):>2}{params.radar_clutter_db:>12g}"
              f"{params.radar_angular_smear_bins:>12g}"
              f"{params.radar_looks:>8g}{stats['floor']:>9.3f}")

    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    OUT_ASSETS.mkdir(parents=True, exist_ok=True)
    OUT_RUNS.mkdir(parents=True, exist_ok=True)

    figure = montage(ladder, panels)
    for suffix in (".pdf", ".png"):
        path = OUT_PAPER / f"fig_radar_ladder{suffix}"
        figure.savefig(path, dpi=300)
        print(f"written       : {path.relative_to(ROOT.parent)}")
    figure.savefig(OUT_RUNS / "radar_operator_check.png", dpi=220)
    plt.close(figure)

    for level, panel in zip(ladder.levels, panels):
        path = OUT_ASSETS / f"阶梯_雷达_s{int(level)}.png"
        Image.fromarray((display(panel) * 255.0).round().astype(np.uint8)).save(path)
    print("written       : 图素材/阶梯_雷达_s0..s4.png   (5 files)")
    print("written       : figures/radar_operator_check.png")
    print()
    print("reminder: the reported runs pass the radar through undegraded;")
    print("          this montage is an operator check, not an input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

