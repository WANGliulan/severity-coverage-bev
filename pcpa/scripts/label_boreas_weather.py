"""Work out and record the road-surface label of each Boreas sequence.

Run from the project root::

    .\\.venv\\Scripts\\python.exe scripts\\label_boreas_weather.py

Boreas ships no weather annotation, so the labels are ours to assign. To keep
that from being guesswork, the script prints for every downloaded sequence the
LiDAR statistics that adverse weather moves --- how much of the sweep survives
out to far range, and how large a share of the returns are weak --- ranks the
sequences by how adverse they look, and shows the season for context.

Then look at the sequence with ``scripts/make_weather_sheets.py``, which tiles
real camera frames into one image per sequence, and write what you see into
``config/boreas_weather.yaml``. Re-running shows what is still unlabelled.

Two caveats, both of them measured rather than assumed:

* The proxy is not a label. It cannot separate rain from snow, it cannot see
  that a drive happened at night, and the far-return fraction tracks how open
  the scene is at least as much as it tracks precipitation --- on the first two
  labelled sequences it ranks them the *opposite* way to what the images show.
* Neither the proxy nor the sheets see the whole drive. A download subsamples
  the sequence, so the stills may cover its opening minute only, and a label
  read off them is a claim about that stretch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config                              # noqa: E402
from src.data.boreas import discover_sequences                         # noqa: E402
from src.data.weather import (                                         # noqa: E402
    WEATHER_VOCABULARY,
    load_weather_map,
    proxy_scores,
    save_weather_map,
    sequence_proxy,
)


def main() -> int:
    cfg = load_config()
    data_root = Path(cfg["paths.data_raw"]) / "boreas"
    bundle = data_root / "boreas-objects-v1"
    sequences = discover_sequences(data_root, bundle)
    if not sequences:
        print(f"no sequences under {data_root}; download some first")
        return 1

    weather_map = load_weather_map()
    rows, labels, names = [], [], []
    for sequence in sequences:
        proxy = sequence_proxy(sequence)
        if not proxy:
            continue
        rows.append(proxy)
        labels.append(weather_map.get(sequence.name, "unlabelled"))
        names.append(sequence.name)

    if not rows:
        print("none of the downloaded sequences has LiDAR frames yet")
        return 1

    scores = proxy_scores(rows)
    order = np.argsort(-scores)          # worst-looking first

    reference_mu = cfg.get_path("surface.friction_mu_by_weather", default={}) or {}
    print(f"{len(names)} sequence(s) with LiDAR frames under {data_root}")
    print(f"vocabulary: {', '.join(WEATHER_VOCABULARY)}\n")
    header = (f"{'sequence':<26}{'season':<7}{'frames':>7}{'pts/frame':>11}"
              f"{'far frac':>10}{'far int':>9}{'low frac':>10}{'proxy':>8}  label")
    print(header)
    print("-" * len(header))
    for index in order:
        name = names[index]
        row = rows[index]
        season = name.replace("boreas-", "")[:4] + "-" + name.replace("boreas-", "")[5:7]
        label = labels[index]
        warning = "" if label in reference_mu else "  (no friction entry)"
        print(f"{name:<26}{season:<7}{row['frames_sampled']:>7.0f}"
              f"{row['points_mean']:>11,.0f}{row['far_fraction']:>10.3f}"
              f"{row['far_intensity']:>9.1f}{row['low_intensity_fraction']:>10.3f}"
              f"{scores[index]:>8.2f}  {label}{warning}")

    unlabelled = [names[i] for i, label in enumerate(labels) if label == "unlabelled"]
    print()
    print("the proxy ranks the worst-looking sweep first. It cannot separate rain")
    print("from snow, and it tracks how open the scene is as much as the weather,")
    print("so look at real frames before setting a label:")
    print("  .\\.venv\\Scripts\\python.exe scripts\\make_weather_sheets.py")
    print("  (figures/weather_sheets/<sequence>.jpg, no video download needed)")
    for name in names:
        print(f"  {name}")
    print()
    if unlabelled:
        print(f"{len(unlabelled)} of {len(names)} sequence(s) still unlabelled: "
              f"{', '.join(unlabelled)}")
        mapping = {name: weather_map.get(name, "unlabelled") for name in names}
        path = save_weather_map(mapping)
        print(f"a template with every sequence marked 'unlabelled' is at {path}")
        print("edit it, then re-run this script to confirm.")
    else:
        print("every downloaded sequence carries a label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
