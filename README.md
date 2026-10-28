# Controlled Severity Coverage Training for Adverse Weather BEV Occupancy Prediction

Code and results for a controlled study of what actually limits a cross-modal
bird's-eye-view (BEV) occupancy stack when the weather turns: the fusion structure, or
the severities the training set contains.

Real Boreas recordings are replayed through a physically parameterized five-level
degradation ladder (`s=0` clear to `s=4` worst), and three structures are trained on
the same split with the same degradation seed: a concatenated-fusion baseline
(117,937 parameters), the same stack with a per-modality reliability gate (262,866
parameters), and an oracle that hands the gate the true severity.

**The structure is not the bottleneck. Training severity is.** A gate with 2.2× the
parameters, even given the true severity, is indistinguishable from concatenation at
the most severe level (0.634 / 0.636 / 0.636 IoU); training on levels 0–2 alone
costs **0.087 IoU** (0.547 → 0.634), 16% relative and four times the run-to-run
spread, and the same effect transfers to detection at 2.3× (0.062 → 0.141 mAP).

## Results

`results/summary.md` is generated from the run records; the numbers below are the
same table, at the two most severe levels.

| trained on | structure | s=3 | s=4 |
|---|---|---:|---:|
| levels 0–2 | concat | 0.663 | **0.547** |
| levels 0–2 | gate | 0.656 | 0.537 |
| levels 0–2 | gate + oracle | 0.654 | 0.534 |
| levels 0–4 | concat | 0.699 | **0.634** |
| levels 0–4 | gate | 0.700 | 0.636 |
| levels 0–4 | gate + oracle | 0.700 | **0.636** |

Detection on the labelled subset, mAP at BEV IoU 0.5 (three runs per regime):

| trained on | structure | s=3 | s=4 |
|---|---|---:|---:|
| levels 0–2 | concat | 0.1383 | 0.0620 |
| levels 0–2 | gate | 0.1227 | 0.0727 |
| levels 0–4 | concat | 0.1537 | 0.1409 |
| levels 0–4 | gate | 0.1546 | 0.1441 |

Every mean is over three or four runs; the per-run values, the spread, and the
coverage effect per level are in `results/tables/`.

## Repository layout

```
pcpa/
  src/degrade/     camera, LiDAR and radar degradation, the five-level ladder
  src/data/        Boreas loader, sequence manifest, weather labels
  src/models/      BEV rasterisation, occupancy network, concatenated and gated
                   fusion, the detection head
  src/eval/        occupancy IoU and BEV detection metrics
  scripts/         data preparation, the two experiments, the results export
  config/          the severity ladder and both experiment configurations (yaml)
  data/manifest/   one JSON record per finished run, plus the sample index
results/
  summary.md       generated tables: occupancy, coverage effect, detection
  tables/*.csv     the same numbers per run, for plotting or re-analysis
  figures/*.png    IoU per level for every run; the ladder on one timestamp;
                   one frame from each of the fourteen drives; the radar
                   operator check (reference only, see below)
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r pcpa/requirements.txt
# torch is installed separately: the models assume CUDA 12.8+ (Blackwell sm_120)
.venv/Scripts/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python pcpa/scripts/check_env.py
```

Then one run per configuration, from the `pcpa` directory:

```bash
python scripts/run_fusion_study.py --train-levels 0,1,2        --seed 0
python scripts/run_fusion_study.py --train-levels 0,1,2,3,4    --seed 0
python scripts/run_detection_study.py --train-levels 0,1,2,3,4 --seed 0
```

Each run writes a record to `data/manifest/`. Regenerate every table from those
records with:

```bash
python scripts/export_results.py
```

## Reproducing a specific number

| what | where | produced by |
|---|---|---|
| IoU per level, every run | `results/tables/fusion_runs.csv` | `run_fusion_study.py` |
| mean per configuration | `results/tables/fusion_mean.csv` | `export_results.py` |
| coverage effect (+0.087) | `results/tables/coverage_effect.csv` | `export_results.py` |
| detection mAP | `results/tables/detection_mean.csv` | `run_detection_study.py` |
| the severity ladder itself | `pcpa/config/default.yaml` | `src/degrade/severity.py` |

## Review-response checks

`results/review_checks.md` collects four controls that a reviewer asked for; each is a
separate run recorded in `pcpa/data/manifest/`, and `pcpa/scripts/analyse_review_experiments.py`
prints them again from the records:

| check | what it answers | result |
|---|---|---|
| capacity match | is the gate's edge just its extra parameters? | a concatenated baseline widened to 264,073 parameters (the gate's own budget) reaches 0.6284 at `s=4`, below the 0.6339 of its narrow version |
| boundary gap | do adjacent frames across the train/test boundary inflate the result? | dropping four frames either side of every drive's boundary leaves the coverage gap intact: 0.5382 against 0.6394 |
| severe level only | is it coverage, or simply more augmented samples? | `s=4` alone reaches 0.6460 at that level, above the full ladder's 0.6339, but collapses at the clean end (0.468 at `s=0`) |
| detection seeds | is the gate-concat difference on detection real? | three seeds per regime: 0.1441 against 0.1409 at `s=4`, +0.003 inside a 0.012 spread across seeds |

## Protocol notes

- The **degradation seed is fixed** across runs, so every configuration sees
  byte-identical inputs and differs only in the training seed. A spread between runs
  is therefore a statement about training, not about the weather.
- Protocol: 40 frames per sequence, 8 epochs, AdamW, stratified 319/139-frame split.
  Records made under any other protocol are skipped by `export_results.py` rather
  than averaged in (`SUPERSEDED` names the ones kept on disk for history).
- A repeated seed is **not** dropped: training is not bit-reproducible here, and two
  runs of the same command reached 0.5558 and 0.5580 at `s=4`. That scatter is the
  quantity the run-to-run comparisons are made against.
- **Only the camera and the LiDAR are degraded.** `build_channels` in
  `run_fusion_study.py` feeds `radar_to_bev(radars[index], ...)`, the raw sweep, so
  the radar enters every model as recorded; radar-only IoU is therefore flat across
  levels by construction, not by a physical claim. The operator does have a radar
  branch (clutter, angular smear, speckle) and `scripts/make_radar_ladder.py` draws
  it, but **no reported number uses it** --- `results/figures/radar_operator_check.png`
  is that drawing and is labelled as a reference.

## Data

The fourteen drives come from the public [Boreas](https://www.boreas.ca/) dataset;
`pcpa/scripts/download_boreas*.ps1` fetch the frames the manifest asks for. Raw
recordings, degraded frames and trained weights are **not** in this repository: they
are tens of gigabytes and reproducible from the scripts. `pcpa/data/manifest/` holds
the sample index and one small JSON per finished run, which is what `results/` is
built from.

## What is not here

The paper source and the compiled manuscript are deliberately excluded. This
repository is the code and the results only.

## License

The code and the derived results in this repository are released under the
[MIT License](LICENSE); copyright (c) 2026 Yawen Tan, Miao Fan, and contributors.

The license covers this repository only. The Boreas recordings are not
redistributed here and stay under the terms of the
[Boreas dataset](https://www.boreas.ca/); downloaded frames, degraded inputs and
trained weights carry those upstream terms.

