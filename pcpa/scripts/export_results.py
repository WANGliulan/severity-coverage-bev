"""Export every reported number as plain CSV, straight from the run records.

Usage::

    python scripts/export_results.py

``data/manifest/`` is the only part of ``data/`` under version control: a run
writes one JSON record there when it finishes, so the result of an experiment is
a diffable artefact rather than a number typed into a table. This script reads
those records back and writes ``results/``: one CSV per table plus a Markdown
summary. Nothing is trained and nothing is recomputed --- the values are the ones
the runs produced.

Records made under a different protocol are skipped rather than averaged in, and
the filter is the same one the figures and tables were made with:

* ``frames_per_sequence`` must be 40 and ``epochs`` 8 (a smoke test uses three
  frames and one epoch);
* the split must be 319/139, the stratified one (an earlier global-tail split
  has 321/137);
* ``SUPERSEDED`` lists records that predate both, kept on disk for the history
  they document but describing an experiment other than the reported one.

A repeated seed is *not* filtered: training is not bit-reproducible here, and two
runs of the same command reached 0.5558 and 0.5580 at severity four. That scatter
is a measurement, not a duplicate.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest"
RESULTS = ROOT.parent / "results"

#: The protocol every reported number uses.
FRAMES_PER_SEQUENCE = 40
EPOCHS = 8
SPLIT = (319, 139)

#: Records from before the stratified split and the final protocol.
SUPERSEDED = frozenset({
    "fusion_study_20261016-132055.json",
    "fusion_study_20261016-133256.json",
    "fusion_study_20261016-155428.json",
    "fusion_study_20261016-155640.json",
    "fusion_study_20261016-160602.json",
    "fusion_study_20261016-160804.json",
    "fusion_study_20261016-162427.json",
    "fusion_study_20261016-163458.json",
    "fusion_study_20261016-194746.json",
})

SEVERITY = (0, 1, 2, 3, 4)

#: Key in a fusion record -> short name used in the CSVs.
FUSION_METHODS = {
    "baseline (concat)": "concat",
    "gated (reliability)": "gate",
    "gated + oracle severity": "gate+oracle",
    # Only the asymmetric records carry these two: a severity is drawn per
    # modality instead of per frame, so one sensor can be degraded while the
    # others stay clear.
    "concat + asymmetric": "concat+asym",
    "gated + asymmetric": "gate+asym",
}

MILD, FULL = "0,1,2", "0,1,2,3,4"


# --------------------------------------------------------------------- loading
def load_fusion() -> list[dict]:
    """Every fusion record on the reported protocol, oldest first."""
    records: list[dict] = []
    for path in sorted(MANIFEST.glob("fusion_study_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        arguments = record.get("args", {})
        if arguments.get("frames_per_sequence") != FRAMES_PER_SEQUENCE:
            continue
        if arguments.get("epochs") != EPOCHS:
            continue
        if path.name in SUPERSEDED:
            continue
        counts = record.get("frames", {})
        if (counts.get("train"), counts.get("test")) != SPLIT:
            continue
        if not record.get("iou"):
            continue
        record["_source"] = path.name
        records.append(record)
    return records


def load_detection() -> list[dict]:
    """Every detection record, in the shape ``run_detection_study.py`` writes.

    Detection records are not filtered on the protocol constants above: their
    split is the labelled subset of one sequence, not the fourteen-sequence split
    the occupancy runs use, and they carry their own ``args`` block.
    """
    records: list[dict] = []
    for path in sorted(MANIFEST.glob("detection_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("results"):
            continue
        record["_source"] = path.name
        records.append(record)
    return records


# -------------------------------------------------------------------- helpers
def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    """Write one table, creating ``results/tables`` on first use."""
    directory = RESULTS / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{path.relative_to(ROOT.parent).as_posix():<44} {len(rows):>4} rows")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def spread(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def levels_of(record: dict, method: str) -> list[float]:
    """One method's ladder from a fusion record."""
    curve = record.get("iou", {}).get(method)
    return [] if curve is None else [float(value) for value in curve]


def regime_of(record: dict) -> str:
    """Training regime of a record, as the comma string the runs were given."""
    return str(record.get("args", {}).get("train_levels"))


def run_id(source: str, prefix: str) -> str:
    """``fusion_study_20261016-211959.json`` -> ``20261016-211959``."""
    return source.removeprefix(prefix).removesuffix(".json")


# ---------------------------------------------------------------------- tables
def fusion_run_rows(records: list[dict]) -> list[list]:
    """One row per (run, structure, level): the raw ladder behind every mean."""
    rows: list[list] = []
    for record in records:
        seed = record.get("args", {}).get("seed")
        asymmetric = bool(record.get("args", {}).get("asymmetric"))
        for method, short in FUSION_METHODS.items():
            for level, value in enumerate(levels_of(record, method)):
                rows.append([run_id(record["_source"], "fusion_study_"), seed,
                             regime_of(record), asymmetric, short, level,
                             f"{value:.6f}"])
    return rows


def fusion_mean_rows(records: list[dict]) -> list[list]:
    """Mean ladder per (regime, structure), the shape of the paper's table."""
    groups: dict[tuple[str, bool], list[dict]] = {}
    for record in records:
        key = (regime_of(record), bool(record.get("args", {}).get("asymmetric")))
        groups.setdefault(key, []).append(record)

    rows: list[list] = []
    for (regime, asymmetric), group in sorted(groups.items()):
        for method, short in FUSION_METHODS.items():
            curves = [curve for curve in (levels_of(record, method)
                                          for record in group) if curve]
            if not curves:
                continue
            per_level = [mean([curve[level] for curve in curves])
                         for level in range(len(SEVERITY))]
            rows.append([regime, asymmetric, short, len(curves),
                         *[f"{value:.4f}" for value in per_level],
                         f"{mean(per_level):.4f}"])
    return rows


def coverage_rows(records: list[dict]) -> list[list]:
    """What training on the mild levels alone costs, per structure and level.

    Restricted to the symmetric protocol: the asymmetric runs draw a severity per
    modality and answer a different question, so they appear in
    ``fusion_mean.csv`` under their own tag instead of being folded in here.
    """
    symmetric = [record for record in records
                 if not bool(record.get("args", {}).get("asymmetric"))]
    rows: list[list] = []
    for method, short in FUSION_METHODS.items():
        mild = [curve for curve in (levels_of(record, method) for record in symmetric
                                    if regime_of(record) == MILD) if curve]
        full = [curve for curve in (levels_of(record, method) for record in symmetric
                                    if regime_of(record) == FULL) if curve]
        if not mild or not full:
            continue
        for level in SEVERITY:
            mild_value = mean([curve[level] for curve in mild])
            full_value = mean([curve[level] for curve in full])
            difference = full_value - mild_value
            relative = 100.0 * difference / mild_value if mild_value else float("nan")
            rows.append([short, level, len(mild), len(full),
                         f"{mild_value:.4f}", f"{full_value:.4f}",
                         f"{difference:+.4f}", f"{relative:+.1f}"])
    return rows


def detection_run_rows(records: list[dict]) -> list[list]:
    """One row per (run, structure, level) on the labelled subset."""
    rows: list[list] = []
    for record in records:
        seed = record.get("args", {}).get("seed")
        regime = ",".join(str(value) for value in record.get("train_levels", []))
        for method, payload in record["results"].items():
            for level in SEVERITY:
                block = payload.get("per_level", {}).get(f"s={level}")
                if block is None:
                    continue
                rows.append([run_id(record["_source"], "detection_"), seed, regime,
                             method, level,
                             f"{block.get('iou', 0.0):.6f}",
                             f"{block.get('map_at_0.5', 0.0):.6f}",
                             f"{block.get('map_at_0.3', 0.0):.6f}",
                             f"{block.get('detections_per_frame', 0.0):.2f}"])
    return rows


def detection_mean_rows(records: list[dict]) -> list[list]:
    """Mean IoU and mAP per (regime, structure, level), with the spread over runs."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        regime = ",".join(str(value) for value in record.get("train_levels", []))
        for method in record["results"]:
            groups.setdefault((regime, method), []).append(record)

    rows: list[list] = []
    for (regime, method), group in sorted(groups.items()):
        for level in SEVERITY:
            iou, mAP, mAP30 = [], [], []
            for record in group:
                block = record["results"][method].get("per_level", {}).get(f"s={level}")
                if block is None:
                    continue
                iou.append(float(block.get("iou", 0.0)))
                mAP.append(float(block.get("map_at_0.5", 0.0)))
                mAP30.append(float(block.get("map_at_0.3", 0.0)))
            if not mAP:
                continue
            rows.append([regime, method, level, len(group),
                         f"{mean(iou):.4f}", f"{mean(mAP):.4f}", f"{spread(mAP):.4f}",
                         f"{mean(mAP30):.4f}"])
    return rows


# --------------------------------------------------------------------- summary
def summary_markdown(fusion_mean: list[list], coverage: list[list],
                     detection_mean: list[list]) -> str:
    """The results summary, generated so it cannot drift from the CSVs."""
    lines = [
        "# Results",
        "",
        "Generated by `pcpa/scripts/export_results.py` from the run records in",
        "`pcpa/data/manifest/`. Every value below is the mean over the runs of a",
        "configuration; the per-run values are in `tables/`.",
        "",
        "## BEV occupancy IoU per severity level",
        "",
        "| trained on | structure | runs | s=0 | s=1 | s=2 | s=3 | s=4 | mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for regime, asymmetric, short, count, *values in fusion_mean:
        tag = f"{regime} (asymmetric)" if asymmetric else regime
        lines.append(f"| {tag} | {short} | {count} | "
                     + " | ".join(str(value) for value in values) + " |")

    lines += [
        "",
        "`concat` is the concatenated-fusion baseline (117,937 parameters), `gate` adds a",
        "per-modality reliability gate (262,866 parameters), and `gate+oracle` hands the",
        "gate the true severity instead of its estimate: the strongest form the mechanism",
        "can take, and not deployable.",
        "",
        "A row tagged `asymmetric` is a different experiment on the same split: the",
        "severity is drawn per modality, so a single sensor can be degraded while the",
        "others stay clear. The `+asym` structures are trained that way; their baselines",
        "are the ordinary `concat` rows.",
        "",
        "## What training on the mild levels alone costs",
        "",
        "| structure | level | levels 0-2 | all levels | difference | relative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for short, level, _mild, _full, mild_value, full_value, difference, relative in coverage:
        lines.append(f"| {short} | {level} | {mild_value} | {full_value} | "
                     f"{difference} | {relative}% |")

    lines += [
        "",
        "## Detection, mAP at BEV IoU 0.5",
        "",
        "| trained on | structure | level | mAP@0.5 | sd over runs | mAP@0.3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for regime, method, level, _count, _iou, mAP, sd, mAP30 in detection_mean:
        lines.append(f"| {regime} | {method} | {level} | {mAP} | {sd} | {mAP30} |")

    lines += [
        "",
        "The detection runs use the labelled subset of a single sequence, so they are a",
        "transfer check rather than a second benchmark.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Write every table and the summary; return a shell exit code."""
    fusion = load_fusion()
    detection = load_detection()

    fusion_mean = fusion_mean_rows(fusion)
    coverage = coverage_rows(fusion)
    detection_mean = detection_mean_rows(detection)

    write_csv("fusion_runs.csv",
              ["run", "seed", "train_levels", "asymmetric", "structure",
               "level", "iou"], fusion_run_rows(fusion))
    write_csv("fusion_mean.csv",
              ["train_levels", "asymmetric", "structure", "runs",
               *[f"s={level}" for level in SEVERITY], "mean"], fusion_mean)
    write_csv("coverage_effect.csv",
              ["structure", "level", "runs_mild", "runs_full", "levels_0_2",
               "all_levels", "difference", "relative_percent"], coverage)
    write_csv("detection_runs.csv",
              ["run", "seed", "train_levels", "structure", "level", "iou",
               "map_at_0.5", "map_at_0.3", "detections_per_frame"],
              detection_run_rows(detection))
    write_csv("detection_mean.csv",
              ["train_levels", "structure", "level", "runs", "iou", "map_at_0.5",
               "sd_map_at_0.5", "map_at_0.3"], detection_mean)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = RESULTS / "summary.md"
    summary.write_text(summary_markdown(fusion_mean, coverage, detection_mean),
                       encoding="utf-8")
    print(f"{summary.relative_to(ROOT.parent).as_posix():<44} "
          f"{len(fusion)} fusion + {len(detection)} detection record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
