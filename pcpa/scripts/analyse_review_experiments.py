"""Read the four review-response experiments back and print one report.

    .\\.venv\\Scripts\\python.exe scripts\\analyse_review_experiments.py

The runs are made by ``scripts/run_review_experiments.ps1``; this script only reads
the records they leave in ``data/manifest`` and compares them against the reported
protocol, so nothing here can drift from a number that was measured:

1. **capacity** --- a concatenated baseline at 264,073 parameters, i.e. the same
   budget as the 262,866-parameter gate (base width 24 instead of 16);
2. **boundary gap** --- the same split with four sampled frames dropped either side
   of every drive's train/test boundary;
3. **severe only** --- trained on ``s=4`` alone, so the sample budget matches the
   mild regime but the diversity does not;
4. **detection seeds** --- a third seed per detection regime.

Writes ``runs/review/report.md`` and prints the same text.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest"
OUT = ROOT / "runs" / "review" / "report.md"

PROTOCOL = {"frames_per_sequence": 40, "epochs": 8, "base_channels": 16}
SPLIT = (319, 139)
MILD, FULL = "0,1,2", "0,1,2,3,4"
BASELINE, GATE, ORACLE = ("baseline (concat)", "gated (reliability)",
                          "gated + oracle severity")


def records() -> list[dict]:
    """Every fusion record, with the fields this report sorts on."""
    out: list[dict] = []
    for path in sorted(MANIFEST.glob("fusion_study_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_source"] = path.name
        record["_args"] = record.get("args", {})
        out.append(record)
    return out


def is_protocol(record: dict) -> bool:
    """The runs the paper reports: base 16, no gap, both regimes, no asymmetric."""
    args = record["_args"]
    if any(args.get(key) != value for key, value in PROTOCOL.items()):
        return False
    if args.get("asymmetric"):
        return False
    counts = record.get("frames", {})
    return (counts.get("train"), counts.get("test")) == SPLIT


def mean_or_none(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def spread(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def level(record: dict, method: str, index: int = 4) -> float | None:
    curve = record.get("iou", {}).get(method)
    return None if curve is None else float(curve[index])


def values_of(records_: list[dict], method: str) -> list[float]:
    return [value for value in (level(record, method) for record in records_)
            if value is not None]


def section_capacity(all_records: list[dict]) -> list[str]:
    """Is the gate's edge over concatenation just its extra parameters?"""
    reported = [r for r in all_records
                if is_protocol(r) and r["_args"]["train_levels"] == FULL]
    matched = [r for r in all_records
               if r["_args"].get("base_channels") == 24
               and r["_args"].get("train_levels") == FULL]
    narrow, gate = values_of(reported, BASELINE), values_of(reported, GATE)
    wide = values_of(matched, BASELINE)
    lines = ["## 1. Capacity match", ""]
    if not wide:
        return lines + ["no capacity-matched run yet", ""]
    lines += [
        "| structure | parameters | runs | level-4 IoU |",
        "|---|---:|---:|---:|",
        f"| concat, base 16 (reported) | 117,937 | {len(narrow)} | {mean_or_none(narrow):.4f} |",
        f"| concat, base 24 (matched) | 264,073 | {len(wide)} | {mean_or_none(wide):.4f} |",
        f"| gate, base 16 (reported) | 262,866 | {len(gate)} | {mean_or_none(gate):.4f} |",
        "",
    ]
    matched_gate = values_of(matched, GATE)
    if matched_gate:
        lines += [f"The gate inside the base-24 runs reaches "
                  f"{mean_or_none(matched_gate):.4f} at {len(matched_gate)} runs; its width "
                  "is larger there, so the matched row is the concatenated one.", ""]
    return lines


def section_gap(all_records: list[dict]) -> list[str]:
    """How much of the result rests on frames either side of the split boundary?"""
    lines = ["## 2. Boundary gap", ""]
    for regime, label in ((MILD, "levels 0-2"), (FULL, "levels 0-4")):
        base = [r for r in all_records
                if is_protocol(r) and r["_args"]["train_levels"] == regime]
        gapped = [r for r in all_records
                  if r["_args"].get("base_channels") == 16
                  and r["_args"].get("boundary_gap", 0) > 0
                  and r["_args"].get("train_levels") == regime]
        clean, cut = values_of(base, BASELINE), values_of(gapped, BASELINE)
        if not cut:
            lines.append(f"- {label}: no gapped run yet")
            continue
        lines.append(f"- {label}: reported split {mean_or_none(clean):.4f} (n={len(clean)}), "
                     f"gap of four frames {mean_or_none(cut):.4f} (n={len(cut)}), "
                     f"change {mean_or_none(cut) - mean_or_none(clean):+.4f}")
    lines.append("")
    return lines


def detection_records() -> dict[str, dict[str, list[float]]]:
    """Detection mAP at s=4 per regime and structure, one entry per run."""
    groups: dict[str, dict[str, list[float]]] = {}
    for path in sorted(MANIFEST.glob("detection_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        regime = ",".join(str(value) for value in record.get("train_levels", []))
        for method, payload in record.get("results", {}).items():
            block = payload.get("per_level", {}).get("s=4")
            if block is None:
                continue
            groups.setdefault(regime, {}).setdefault(method, []).append(
                float(block.get("map_at_0.5", 0.0)))
    return groups


def section_detection() -> list[str]:
    """The gate-concat difference with three seeds instead of two."""
    groups = detection_records()
    lines = ["## 4. Detection, mAP at s=4", "",
             "| trained on | structure | runs | mAP@0.5 | sd |",
             "|---|---|---:|---:|---:|"]
    for regime in (MILD, FULL):
        for method, values in sorted(groups.get(regime, {}).items()):
            lines.append(f"| {regime} | {method} | {len(values)} | "
                         f"{mean_or_none(values):.4f} | {spread(values):.4f} |")
    lines.append("")
    for regime in (MILD, FULL):
        block = groups.get(regime, {})
        if "concat" in block and "gate" in block \
                and len(block["concat"]) == len(block["gate"]):
            paired = [gate - concat for concat, gate in zip(block["concat"], block["gate"])]
            lines.append(f"- {regime}: gate minus concat {mean_or_none(paired):+.4f} "
                         f"(paired, n={len(paired)}, range {min(paired):+.4f}"
                         f"..{max(paired):+.4f})")
    lines.append("")
    return lines


def main() -> int:
    all_records = records()
    text = "\n".join(["# Review-response experiments", ""]
                     + section_capacity(all_records)
                     + section_gap(all_records)
                     + section_severe_only(all_records)
                     + section_detection())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


def section_severe_only(all_records: list[dict]) -> list[str]:
    """Coverage, or simply more augmented samples?"""
    severe = [r for r in all_records if r["_args"].get("train_levels") == "4"]
    lines = ["## 3. Severe level only (no diversity, same sample budget)", ""]
    if not severe:
        return lines + ["no severe-only run yet", ""]
    mild = [r for r in all_records
            if is_protocol(r) and r["_args"]["train_levels"] == MILD]
    full = [r for r in all_records
            if is_protocol(r) and r["_args"]["train_levels"] == FULL]
    lines += [
        "| training set | level-4 IoU (concat) |",
        "|---|---:|",
        f"| levels 0-2 (reported) | {mean_or_none(values_of(mild, BASELINE)):.4f} |",
        f"| level 4 only | {mean_or_none(values_of(severe, BASELINE)):.4f} |",
        f"| levels 0-4 (reported) | {mean_or_none(values_of(full, BASELINE)):.4f} |",
        "",
        "If s=4 alone lands near the 0-4 row, severity exposure is what moved the number",
        "and diversity is not needed to explain it; if it lands at or below the 0-2 row,",
        "the worst level alone is not enough and the whole ladder matters.",
        "",
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())

