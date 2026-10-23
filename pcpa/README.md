# pcpa — perception under adverse weather

The code behind the results in `../results`. One five-level degradation ladder, three
fusion structures, two experiments (occupancy and detection), all on real Boreas
recordings.

## Environment

| item | value | why |
|---|---|---|
| interpreter | Python 3.13 | 3.14 ships torch but opencv / scikit-learn wheels lag |
| torch | cu128 (2.11) | the RTX 5070 is Blackwell sm_120; older builds raise `no kernel image is available` |
| GPU memory | 8 GB | render the degraded frames first, then train; do not do both at once |
| detection | torchvision's own models | `mmdetection` / `mmcv` have no cp313 wheels |

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe scripts\check_env.py
```

## Scripts

| script | what it does |
|---|---|
| `check_env.py` | versions, GPU, the sm_120 kernel check |
| `download_boreas.ps1`, `download_boreas_batch.ps1` | fetch the frames the manifest asks for |
| `inspect_boreas.py`, `frame_coverage.py` | what a download actually covers, per sensor |
| `prune_unaligned.py` | move aside frames from the earlier unaligned sampling |
| `label_boreas_weather.py` | assign each sequence a weather label from LiDAR statistics |
| `per_modality_retention.py` | one model per modality, to see what each sensor retains |
| `run_fusion_study.py` | the main experiment: three structures × two training regimes |
| `run_detection_study.py` | the same ladder on the labelled subset, mAP per severity |
| `run_campaign.py` | the full campaign as a sequence of independent steps |
| `export_results.py` | every reported table as CSV, read back from `data/manifest/` |
| `demo_*.py` | smoke tests for the loader, the ladder, BEV rasterisation, the metrics |

## Layout

```
src/common/config.py     yaml loading, no globals
src/degrade/             camera, lidar, radar operators and the severity ladder
src/data/                Boreas loader, sequence manifest, weather labels
src/models/              bev, network (occupancy), gated (fusion), detection
src/eval/                occupancy IoU, BEV detection metrics
config/default.yaml      the ladder every reported number uses
config/boreas_*.yaml     sequence and weather configuration
data/manifest/           sample index + one JSON per finished run
```

## Two rules the code follows

1. **One degradation seed for the whole campaign.** Every configuration then sees
   byte-identical inputs and differs only in the training seed, so a spread between
   runs measures training rather than the weather.
2. **A run writes its own record.** `run_fusion_study.py` and
   `run_detection_study.py` append a JSON under `data/manifest/`; every table in
   `../results` is read back from those files, so no number is ever transcribed by
   hand.
