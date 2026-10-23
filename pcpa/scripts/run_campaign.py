"""Run the experiment campaign the manuscript reports, one step at a time.

    .\\.venv\\Scripts\\python.exe scripts\\run_campaign.py

Four fusion runs --- two training regimes, two training seeds --- followed by the
per-modality retention study. Each step is a separate subprocess with its own log
under ``D:\\``, so a failure in one step cannot hide the others and the logs can be
read while the campaign is still running.

Two properties are deliberate. The degradation seed is fixed across every step,
so all runs see byte-identical inputs and differ only in the training seed, which
is what makes a spread between runs a statement about training rather than about
noise in the data. And every rasterised stack is cached by
``run_fusion_study.channels_for_level``, so only the first step pays for the
rasterisation and the rest start immediately.
"""

from __future__ import annotations

import atexit
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

#: A second lock, for the launcher rather than for a run. The per-run lock in
#: ``run_fusion_study`` stops two *experiments* from interleaving, but it cannot
#: stop two *drivers*: the second one simply waits for the lock, then repeats work
#: whose result is already in the manifest. That happened once, and the cost was a
#: duplicated run whose seed then had to be discounted by hand. Refusing to start
#: is the fix, and a marker directory is atomic the same way the run lock is.
DRIVER_LOCK = Path(tempfile.gettempdir()) / "pcpa_campaign.lock"
MAX_DRIVER_AGE_SECONDS = 1800

#: (name, command-line arguments, log file). Kept short and ASCII: the log paths
#: are written by the caller's shell, which handles the non-ASCII repository path.
STEPS: list[tuple[str, list[str], str]] = [
    ("fusion mild seed 0",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "0"],
     r"D:\campaign_mild_s0.log"),
    ("fusion full seed 0",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "0", "--train-levels", "0,1,2,3,4"],
     r"D:\campaign_full_s0.log"),
    ("fusion mild seed 1",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "1"],
     r"D:\campaign_mild_s1.log"),
    ("fusion full seed 1",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "1", "--train-levels", "0,1,2,3,4"],
     r"D:\campaign_full_s1.log"),
    ("per-modality retention",
     ["scripts/per_modality_retention.py", "--frames-per-sequence", "40",
      "--epochs", "8"],
     r"D:\campaign_retention.log"),
    # The asymmetric attempt is a negative result, but it is reported next to the
    # symmetric tables, so it has to be measured on the same stratified split as
    # they are; an older run of it under the previous split is superseded.
    ("asymmetric seed 0",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "0", "--asymmetric", "--asym-draws", "2"],
     r"D:\campaign_asym_s0.log"),
    ("wave2 asymmetric seed 1",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "1", "--asymmetric", "--asym-draws", "2"],
     r"D:\campaign_asym_s1.log"),
    # A third seed for the headline comparison, and two more for the retention
    # study, so the main claim rests on three runs rather than two and the
    # per-modality table is not a single draw.
    ("wave2 mild seed 2",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "2"],
     r"D:\campaign_mild_s2.log"),
    ("wave3 full seed 2",
     ["scripts/run_fusion_study.py", "--frames-per-sequence", "40", "--epochs", "8",
      "--seed", "2", "--train-levels", "0,1,2,3,4"],
     r"D:\campaign_full_s2.log"),
    ("wave3 retention seed 1",
     ["scripts/per_modality_retention.py", "--frames-per-sequence", "40",
      "--epochs", "8", "--seed", "1"],
     r"D:\campaign_retention_s1.log"),
    ("wave3 retention seed 2",
     ["scripts/per_modality_retention.py", "--frames-per-sequence", "40",
      "--epochs", "8", "--seed", "2"],
     r"D:\campaign_retention_s2.log"),
    # The detection task needs labels, which exist for one sequence and 279 usable
    # frames, and its head converges more slowly than the occupancy head, so the
    # schedule is longer. Both regimes are run so that the detection result can be
    # read the same way as the occupancy one.
    ("wave4 detection mild",
     ["scripts/run_detection_study.py", "--epochs", "32", "--seed", "0"],
     r"D:\campaign_detect_mild.log"),
    ("wave4 detection full",
     ["scripts/run_detection_study.py", "--epochs", "32", "--seed", "0",
      "--train-levels", "0,1,2,3,4"],
     r"D:\campaign_detect_full.log"),
    # A second seed for each detection regime, so the two claims in Section 6.8
    # ("leads by 0.047 while the distribution is narrow", "0.003 once it is not")
    # rest on a spread rather than on one run.
    ("wave5 detection mild seed 1",
     ["scripts/run_detection_study.py", "--epochs", "32", "--seed", "1"],
     r"D:\campaign_detect_mild_s1.log"),
    ("wave5 detection full seed 1",
     ["scripts/run_detection_study.py", "--epochs", "32", "--seed", "1",
      "--train-levels", "0,1,2,3,4"],
     r"D:\campaign_detect_full_s1.log"),
]


def acquire_driver_lock() -> bool:
    """Take the launcher lock, so two drivers cannot repeat each other's work."""
    if DRIVER_LOCK.exists():
        age = time.time() - DRIVER_LOCK.stat().st_mtime
        if age < MAX_DRIVER_AGE_SECONDS:
            print(f"another campaign driver holds {DRIVER_LOCK} "
                  f"({age / 60:.0f} min old); exiting. Delete that directory if no "
                  f"driver is running.")
            return False
        print(f"removing a stale campaign lock ({age / 60:.0f} min old)")
        DRIVER_LOCK.rmdir()
    # mkdir without exist_ok is atomic, so exactly one driver can win the race.
    try:
        DRIVER_LOCK.mkdir(parents=True)
    except FileExistsError:
        print(f"another campaign driver took {DRIVER_LOCK}; exiting")
        return False
    return True


def release_driver_lock() -> None:
    """Drop the launcher lock, tolerating the case where it is already gone."""
    if DRIVER_LOCK.exists():
        DRIVER_LOCK.rmdir()


def main() -> int:
    if not PYTHON.is_file():
        print(f"missing interpreter: {PYTHON}")
        return 1
    if not acquire_driver_lock():
        return 1
    atexit.register(release_driver_lock)
    # An optional substring selects a subset of the steps, so a single stage can be
    # re-run without repeating the whole campaign.
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for index, (name, arguments, log) in enumerate(STEPS, start=1):
        if only and only not in name:
            continue
        print(f"[{index}/{len(STEPS)}] {name} -> {log}", flush=True)
        started = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as handle:
            completed = subprocess.run([str(PYTHON), "-u", *arguments],
                                       cwd=str(ROOT), stdout=handle,
                                       stderr=subprocess.STDOUT, check=False)
        print(f"    exit {completed.returncode} in {time.time() - started:.0f} s",
              flush=True)
    print("campaign finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
