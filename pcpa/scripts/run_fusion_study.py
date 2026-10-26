"""Does reliability gating beat plain concatenation, under degradation?

    .\\.venv\\Scripts\\python.exe scripts\\run_fusion_study.py --frames-per-sequence 60

Two models see exactly the same data, on the same frames, with the same budget:

  baseline  : :class:`~src.models.network.BevOccupancyNet` on the three modality
              planes concatenated as input channels -- the current configuration.
  gated     : :class:`~src.models.gated.GatedFusionNet`, one stream per modality
              plus a learned (modality x severity) reliability table that weights
              the streams, supervised only through the severity heads.

Training uses severity levels 0--2 and evaluation runs over 0--4, so both models
are asked to generalise to degradation they never saw. The comparison is reported
as BEV IoU per level, together with the learned reliability table, which is the
readable part of the result: it should recover the ordering that the retention
curve measures directly (camera least reliable, LiDAR most).

The frame set and the degradation operators are the ones the A2 ablation uses --
``run_a2_ablation`` is imported rather than copied so the two studies cannot drift
apart. What is deliberately absent is the contact and memory plane: this study is
about cross-modal fusion among the modalities that the weather attacks.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Results live beside the other manifests, which are the only part of data/ that
#: is under version control, so a run leaves a diffable artifact behind.
MANIFEST_DIR = ROOT / "data" / "manifest"

import torch                                                        # noqa: E402

import run_a2_ablation as a2                                        # noqa: E402
from src.common.config import load_config                           # noqa: E402
from src.degrade import camera as cam                               # noqa: E402
from src.degrade import lidar as lid                                # noqa: E402
from src.degrade import radar as rad                                # noqa: E402
from src.degrade.severity import SeverityLadder                     # noqa: E402
from src.models.bev import (BevGrid, camera_to_bev_batch,           # noqa: E402
                            lidar_to_bev, occupancy_reference, radar_to_bev)
from src.models.gated import (GatedFusionNet, evaluate_gated,       # noqa: E402
                              fit_gated, gate_table, mean_gates)
from src.models.network import (BevOccupancyNet, TrainConfig,       # noqa: E402
                                evaluate, fit, input_channels)

#: The modalities this study fuses. Contact and memory are out of scope by design.
MODALITIES = ("camera", "lidar", "radar")
TRAIN_LEVELS = (0, 1, 2)
TEST_LEVELS = (0, 1, 2, 3, 4)


def as_batch(block: np.ndarray) -> np.ndarray:
    """Coerce one modality's stacked frames to ``(N, C, H, W)``.

    The rasterisers disagree about their return shape -- ``lidar_to_bev`` yields a
    channel stack, ``radar_to_bev`` can yield a bare plane -- and the shape is not
    worth guessing at, so it is normalised here and printed by the caller.
    """
    if block.ndim == 3:            # (N, H, W)
        return block[:, None]
    if block.ndim == 2:            # (H, W), a single frame
        return block[None, None]
    return block


def build_channels(cfg, ladder, scenes, cameras, radars, camera_calib, grid, levels, seed):
    """BEV stacks per severity level, built exactly as the A2 ablation builds them.

    Same operators, same parameters, same rasterisation, so the only difference
    between this study and the A2 study is which model consumes the planes.
    """
    ipm_range = float(cfg.get_path("degradation.camera.ipm_max_range_m", 40.0))
    rng = np.random.default_rng(seed)
    channels: dict[int, dict[str, np.ndarray]] = {}
    for level in levels:
        params = ladder.at(level).as_dict()
        lidar_stack, camera_stack, radar_stack = [], [], []
        for index in range(len(scenes)):
            degraded = lid.degrade(scenes[index], lid.LidarDegradation(
                extinction_beta_per_m=params["lidar_beta_per_m"],
                spurious_return_rate=params["lidar_spurious_rate"],
                beam_dilation_rad=params["lidar_beam_dilation_rad"],
                range_noise_frac=params["lidar_range_noise_frac"]), rng)
            lidar_stack.append(lidar_to_bev(degraded, grid))
            degraded_image = cam.degrade(cameras[index], cam.CameraDegradation(
                extinction_beta_per_m=params["camera_beta_per_m"],
                streak_density=params["camera_streak_density"],
                exposure_loss=params["camera_exposure_loss"],
                read_noise_std=params["camera_read_noise_std"],
                scene_depth_m=float(cfg["degradation.camera.scene_depth_m"]),
                atmospheric_light=float(cfg["degradation.camera.atmospheric_light"])), rng)
            camera_stack.append(camera_to_bev_batch(degraded_image[None, ...], camera_calib,
                                                    grid, max_range_m=ipm_range)[0])
            radar_stack.append(radar_to_bev(radars[index], grid, max_range_m=100.0)[0])
        channels[level] = {
            name: as_batch(np.stack(stack)) for name, stack in
            (("lidar", lidar_stack), ("camera", camera_stack), ("radar", radar_stack))
        }
        if level == levels[0]:
            print("  channel shapes: " + ", ".join(
                f"{name}={tuple(channels[level][name].shape)}" for name in sorted(channels[level])))
        coverage = float(np.mean(channels[level]["camera"][:, 1] > 0.0))
        print(f"  s={level}: {len(scenes)} frames rasterised, "
              f"camera footprint covers {coverage * 100:.1f}% of the grid")
    return channels


def stacked(channels, level: int, modalities: tuple[str, ...]) -> np.ndarray:
    """The input tensor for one severity level and one modality subset."""
    return np.concatenate([channels[level][modality] for modality in modalities], axis=1)


def asymmetric_stacks(channels, modality_levels, modalities: tuple[str, ...],
                      frame_index=None) -> np.ndarray:
    """One input per sample, each with a *different* severity per modality.

    The ladder is applied modality by modality, so a training sample can mix
    levels: the camera ruined while the LiDAR is intact, or the reverse. Because
    every level is generated from the same clean frame, this costs no new
    recording, and it is the only configuration in which the gate can observe that
    one modality has failed on its own and therefore has to be discounted.

    ``modality_levels`` is an ``(N, M)`` array: row ``i`` says which level each
    modality takes for sample ``i``. ``frame_index`` selects which frames of the
    stack the samples are drawn from, which is what the per-sequence split needs.
    """
    levels = np.asarray(modality_levels, dtype=np.int64)
    count = levels.shape[0]
    frames = slice(None) if frame_index is None else np.asarray(frame_index)
    # Start from the level of the first row, then overwrite the rows that asked
    # for something else. Only the distinct levels touch the data, so this is a
    # handful of fancy-index assignments rather than a Python loop per sample.
    planes = [channels[int(levels[0, index])][name][frames][:count]
              for index, name in enumerate(modalities)]
    for index, name in enumerate(modalities):
        for level in np.unique(levels[:, index]):
            rows = np.nonzero(levels[:, index] == level)[0]
            planes[index][rows] = channels[int(level)][name][frames][:count][rows]
    return np.concatenate(planes, axis=1)


#: Fraction of each *sequence* that goes to training. Splitting per sequence
#: rather than across the whole frame list matters more than it looks: a single
#: global cut puts whole drives on one side of the split, and since the frames are
#: grouped by sequence, that silently changes what the test set contains. An
#: earlier version of this split left both rain sequences in training and put 37 %
#: of the evaluation on clear-weather frames, which makes any per-condition reading
#: meaningless and lets the model be evaluated on scenes it has never seen.
TRAIN_FRACTION = 0.7


def stratify_by_sequence(names, fraction: float = TRAIN_FRACTION, gap: int = 0):
    """Split frame indices per sequence, so every condition appears on both sides.

    Sequences of one frame are kept whole in training; every longer sequence
    contributes at least one frame to evaluation. ``gap`` drops that many frames
    either side of the boundary, which is what the leakage check in
    ``scripts/run_review_experiments.ps1`` uses: adjacent frames of one drive are
    correlated, so a gap measures how much of a result survives when the two sides
    cannot see each other. The reported protocol uses ``gap = 0``.
    """
    names = np.asarray(names)
    train: list[int] = []
    test: list[int] = []
    for name in dict.fromkeys(names.tolist()):
        index = np.nonzero(names == name)[0]
        if index.size < 2:
            train.extend(index.tolist())
            continue
        cut = min(max(1, int(round(fraction * index.size))), index.size - 1)
        # A short drive cannot afford the full gap: with 12 frames, 70/30 leaves four
        # on the test side, and dropping four there would empty it. The gap is
        # therefore clamped per sequence, so a drive either contributes a gap or keeps
        # its boundary --- never a half-empty side.
        gap_here = min(gap, cut - 1, index.size - cut - 1)
        keep_train = index[:cut - gap_here]
        keep_test = index[cut + gap_here:]
        train.extend(keep_train.tolist())
        test.extend(keep_test.tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


LOCK_PATH = Path(tempfile.gettempdir()) / "pcpa_fusion_study.lock"
MAX_LOCK_AGE_SECONDS = 1800


def acquire_lock() -> bool:
    """Take an exclusive lock, so two studies cannot interleave into one log.

    Two runs of the same experiment writing the same file produce a report that
    belongs to neither, so the second one refuses to start instead. An abandoned
    lock is ignored after ``MAX_LOCK_AGE_SECONDS``.
    """
    if LOCK_PATH.exists():
        age = time.time() - LOCK_PATH.stat().st_mtime
        if age < MAX_LOCK_AGE_SECONDS:
            print(f"another run holds {LOCK_PATH} ({age / 60:.0f} min old); exiting")
            return False
        print(f"removing a stale lock ({age / 60:.0f} min old)")
        LOCK_PATH.rmdir()
    try:
        # mkdir without exist_ok is atomic, which is what makes the lock race-free:
        # two processes that check-and-create simultaneously cannot both win.
        LOCK_PATH.mkdir(parents=True)
    except FileExistsError:
        print(f"another run took {LOCK_PATH} while this one was starting; exiting")
        return False
    return True


def release_lock() -> None:
    """Drop the lock, tolerating the case where it is already gone."""
    if LOCK_PATH.exists():
        LOCK_PATH.rmdir()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames-per-sequence", type=int, default=40,
                        help="frames taken per Boreas sequence")
    parser.add_argument("--stride", type=int, default=1,
                        help="step through the *already downloaded* frames of each sequence; "
                             "keep this at 1 because the downloader already subsampled them")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16,
                        help="width of the baseline and of each gated stream")
    parser.add_argument("--gated-base-channels", type=int, default=None,
                        help="override the gated stream width, to match parameter counts")
    parser.add_argument("--aux-weight", type=float, default=0.5,
                        help="weight on the severity-classification auxiliary loss")
    parser.add_argument("--train-levels", default="0,1,2",
                        help="severities mixed into training; the default reproduces the A2 "
                             "split, while '0,1,2,3,4' measures the upper bound of having "
                             "seen every level during training")
    parser.add_argument("--asymmetric", action="store_true",
                        help="the method: draw a severity per modality instead of one per "
                             "sample, so the training set contains frames in which a single "
                             "modality has failed while the others are clean")
    parser.add_argument("--asym-draws", type=int, default=2,
                        help="asymmetric draws per training frame, i.e. the data budget knob")
    parser.add_argument("--degradation-seed", type=int, default=0,
                        help="seed for the degradation operators, kept separate from the "
                             "training seed so that every run sees identical inputs")
    parser.add_argument("--boundary-gap", type=int, default=0,
                        help="frames dropped either side of each drive's train/test "
                             "boundary; 0 is the reported protocol, and a positive value "
                             "measures how much of a result rests on adjacent frames")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


#: Rasterised stacks are cached on disk. Building five levels of 458 frames takes
#: minutes, and an experiment campaign repeats exactly that work for every run, so
#: the cache is what makes a batch of runs affordable. The key carries the frame
#: set, the severity, the seed the degradation used, and the cache is discarded
#: rather than trusted if it cannot be read back.
CACHE = ROOT / "data" / "cache" / "bev"


def cache_key(sequence_names, frames_per_sequence: int, stride: int, level: int,
              seed: int) -> str:
    return (f"{len(sequence_names)}seq-{frames_per_sequence}f-{stride}s-"
            f"s{level}-seed{seed}-{sequence_names[0]}-{sequence_names[-1]}")


def channels_for_level(cfg, ladder, scenes, cameras, radars, camera_calib, grid,
                       level: int, seed: int, sequence_names,
                       frames_per_sequence: int, stride: int):
    """Channels for one severity level, from the cache when it is usable."""
    cache = CACHE / f"{cache_key(sequence_names, frames_per_sequence, stride, level, seed)}.npz"
    if cache.is_file():
        try:
            with np.load(cache) as data:
                return {name: data[name] for name in data.files}
        except Exception as error:                       # noqa: BLE001
            print(f"  cache {cache.name} unreadable ({error}); rebuilding")
    built = build_channels(cfg, ladder, scenes, cameras, radars, camera_calib,
                           grid, (level,), seed)[level]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **built)
    print(f"  cached {cache.name}")
    return built


def parameter_count(model) -> int:
    """Trainable parameters, printed so the comparison is honest about capacity."""
    return int(sum(parameter.numel() for parameter in model.parameters()
                   if parameter.requires_grad))


def per_frame_iou(model, inputs: np.ndarray, targets: np.ndarray,
                  gated: bool) -> np.ndarray:
    """IoU of every frame separately, so that frames can be grouped afterwards.

    The aggregate helpers return one number for a whole set, which is what the
    main tables need. A breakdown by weather condition needs one number per frame
    and that is all this is: a loop of single-frame evaluations.
    """
    out = np.empty(inputs.shape[0], dtype=float)
    for index in range(inputs.shape[0]):
        single = inputs[index:index + 1]
        reference = targets[index:index + 1]
        out[index] = (evaluate_gated(model, single, reference) if gated
                      else evaluate(model, single, reference))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not acquire_lock():
        return 1
    atexit.register(release_lock)
    train_levels = tuple(int(value) for value in str(args.train_levels).split(","))
    if train_levels == TRAIN_LEVELS:
        # Only the default path claims to reproduce the A2 split; an explicit
        # --train-levels is a deliberate change of regime, not a drift.
        assert tuple(a2.TRAIN_LEVELS) == TRAIN_LEVELS, "level split drifted from the A2 study"

    cfg = load_config()
    grid = BevGrid()
    ladder = SeverityLadder.from_config(cfg)
    camera_shape = (300, 400)
    camera_calib = a2.camera_calibration_or_default(cfg, camera_shape)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"source   : real Boreas recordings, {args.frames_per_sequence} frames/sequence, "
          f"stride {args.stride}")
    print(f"device   : {device}")
    print(f"severity : train on {train_levels}, evaluate on {TEST_LEVELS}\n")

    scenes, cameras, radars, _, sequence_names = a2.build_real_frames(
        cfg, camera_shape, args.frames_per_sequence, args.stride)
    if len(scenes) < 6:
        print("not enough aligned frames; download more sequences first")
        return 1
    total = len(scenes)
    targets = np.stack([occupancy_reference(scene, grid) for scene in scenes])
    train_index, test_index = stratify_by_sequence(sequence_names, TRAIN_FRACTION,
                                                   args.boundary_gap)
    train_count = int(train_index.size)
    test = test_index
    print(f"\nframes   : {total} ({train_count} train / {test_index.size} test), "
          f"split per sequence"
          + (f", boundary gap {args.boundary_gap}" if args.boundary_gap else ""))

    print("\nbuilding the degraded BEV stacks")
    channels = {level: channels_for_level(cfg, ladder, scenes, cameras, radars,
                                          camera_calib, grid, level,
                                          args.degradation_seed,
                                          sequence_names, args.frames_per_sequence,
                                          args.stride)
                for level in TEST_LEVELS}

    train_x = np.concatenate([stacked(channels, level, MODALITIES)[train_index]
                              for level in train_levels], axis=0)
    train_y = np.concatenate([targets[train_index]] * len(train_levels), axis=0)
    level_labels = np.concatenate(
        [np.full(train_count, level, dtype=np.int64) for level in train_levels])
    config = TrainConfig(epochs=args.epochs, device=device, seed=args.seed,
                         auxiliary_weight=args.aux_weight)
    validation = (stacked(channels, train_levels[-1], MODALITIES)[test], targets[test])

    # ---- the method: one severity per modality -----------------------
    asymmetric = None
    if args.asymmetric:
        rng = np.random.default_rng(args.seed)
        pool = np.asarray(train_levels, dtype=np.int64)
        draws_x, draws_y, draws_levels = [], [], []
        for _ in range(max(1, args.asym_draws)):
            choices = rng.choice(pool, size=(train_count, len(MODALITIES)))
            draws_x.append(asymmetric_stacks(channels, choices, MODALITIES,
                                             train_index))
            draws_y.append(targets[train_index])
            draws_levels.append(choices)
        asymmetric = (np.concatenate(draws_x, axis=0),
                      np.concatenate(draws_y, axis=0),
                      np.concatenate(draws_levels, axis=0))
        print(f"\nasymmetric augmentation: {args.asym_draws} draw(s) x {train_count} "
              f"frames, one severity per modality, drawn from "
              f"{tuple(int(value) for value in pool)}")
        for index, name in enumerate(MODALITIES):
            histogram = np.bincount(asymmetric[2][:, index],
                                    minlength=len(TEST_LEVELS)).tolist()
            print(f"  {name:<8} levels {histogram}")

    results: dict[str, list[float]] = {}
    started = time.time()

    baseline = BevOccupancyNet(input_channels(MODALITIES), args.base_channels)
    print(f"\nbaseline  (concatenated channels, {parameter_count(baseline)} parameters)")
    fit(baseline, train_x, train_y, config)
    results["baseline (concat)"] = [
        evaluate(baseline, stacked(channels, level, MODALITIES)[test], targets[test])
        for level in TEST_LEVELS]
    print("  " + "  ".join(f"s={level}:{value:.4f}"
                           for level, value in zip(TEST_LEVELS, results["baseline (concat)"])))

    if asymmetric is not None:
        # The same augmentation handed to concatenation, so that a gain of the
        # gate can be told apart from a gain of the augmentation itself.
        asymmetric_baseline = BevOccupancyNet(input_channels(MODALITIES),
                                              args.base_channels)
        print(f"\nbaseline  (concat, asymmetric augmentation, "
              f"{parameter_count(asymmetric_baseline)} parameters)")
        fit(asymmetric_baseline, asymmetric[0], asymmetric[1], config)
        results["concat + asymmetric"] = [
            evaluate(asymmetric_baseline, stacked(channels, level, MODALITIES)[test],
                     targets[test])
            for level in TEST_LEVELS]
        print("  " + "  ".join(f"s={level}:{value:.4f}" for level, value in
                               zip(TEST_LEVELS, results["concat + asymmetric"])))

    gated_base = args.gated_base_channels or args.base_channels
    gated = GatedFusionNet(MODALITIES, num_levels=len(TEST_LEVELS), base_channels=gated_base)
    print(f"\ngated     (one stream per modality + reliability table, "
          f"{parameter_count(gated)} parameters)")
    gated_key = "gated + asymmetric" if asymmetric is not None else "gated (reliability)"
    if asymmetric is not None:
        fit_gated(gated, asymmetric[0], asymmetric[1], asymmetric[2], config,
                  validation=validation)
    else:
        fit_gated(gated, train_x, train_y, level_labels, config, validation=validation)
    results[gated_key] = [
        evaluate_gated(gated, stacked(channels, level, MODALITIES)[test], targets[test])
        for level in TEST_LEVELS]
    print("  " + "  ".join(f"s={level}:{value:.4f}"
                           for level, value in zip(TEST_LEVELS, results[gated_key])))

    # Oracle severity: the reliability table receives the *true* level instead of
    # the level the heads predicted. This bounds what any severity estimator can
    # buy on top of the gate, so it decides whether the estimation and adaptation
    # route has room at all -- and by how much.
    results["gated + oracle severity"] = [
        evaluate_gated(gated, stacked(channels, level, MODALITIES)[test], targets[test],
                       severity_override=level)
        for level in TEST_LEVELS]
    print("  " + "  ".join(f"s={level}:{value:.4f}"
                           for level, value in zip(TEST_LEVELS,
                                                   results["gated + oracle severity"])))

    # ---- report ------------------------------------------------------
    print("\nBEV IoU per severity level (higher is better)")
    print(f"{'configuration':<22}" + "".join(f"{'s=' + str(level):>9}" for level in TEST_LEVELS))
    print("-" * (22 + 9 * len(TEST_LEVELS)))
    for name, values in results.items():
        print(f"{name:<22}" + "".join(f"{value:>9.4f}" for value in values))
    deltas = [g - b for g, b in zip(results[gated_key],
                                    results["baseline (concat)"])]
    print(f"{'delta (gated - base)':<22}" + "".join(f"{value:>+9.4f}" for value in deltas))

    print("\nlearned reliability table (row = modality, column = severity)")
    print(f"{'modality':<10}" + "".join(f"{'s=' + str(level):>9}" for level in TEST_LEVELS))
    table = gate_table(gated)
    for name, row in table.items():
        print(f"{name:<10}" + "".join(f"{value:>+9.3f}" for value in row))

    print("\nmean gate actually applied on the test frames")
    print(f"{'level':<8}" + "".join(f"{name:>10}" for name in MODALITIES))
    for level in TEST_LEVELS:
        gates = mean_gates(gated, stacked(channels, level, MODALITIES)[test])
        print(f"{'s=' + str(level):<8}" + "".join(f"{gates[name]:>10.3f}" for name in MODALITIES))

    print("\nmean gate with the true severity (oracle upper bound)")
    print(f"{'level':<8}" + "".join(f"{name:>10}" for name in MODALITIES))
    severity_shift: dict[str, float] = {}
    for level in TEST_LEVELS:
        inputs = stacked(channels, level, MODALITIES)[test]
        predicted = mean_gates(gated, inputs)
        oracle_gates = mean_gates(gated, inputs, severity_override=level)
        # Total variation between the two gate distributions: how far the severity
        # path moves the gate at all. Near zero would mean the gate is driven
        # entirely by the feature heads and the severity table is decorative,
        # which is the failure mode an estimation-and-adaptation method must avoid.
        severity_shift[str(level)] = 0.5 * sum(abs(predicted[name] - oracle_gates[name])
                                              for name in MODALITIES)
        print(f"{'s=' + str(level):<8}" + "".join(f"{oracle_gates[name]:>10.3f}"
                                                  for name in MODALITIES))
    print("gate shift from knowing the severity: "
          + "  ".join(f"s={level}:{value:.3f}" for level, value in severity_shift.items()))

    # ---- breakdown by the weather that was actually recorded ----------
    # The frames come from real drives, so the same ladder can be reported per
    # condition. This is the table that shows whether a snow sequence behaves
    # differently from a dry one, and it needs no annotation for the same reason
    # the main metric does not: the reference is generated by the LiDAR itself.
    from src.data.weather import load_conditions

    condition_map = load_conditions()
    test_names = [sequence_names[index] for index in test_index]
    condition_of = np.asarray([condition_map.get(name, "unlabelled")
                               for name in test_names])
    print(f"\nBEV IoU by real weather condition ({len(test_names)} test frames), "
          f"concatenated baseline trained on {train_levels}")
    print(f"{'condition':<12}{'frames':>7}"
          + "".join(f"{'s=' + str(level):>9}" for level in TEST_LEVELS))
    iou_by_condition: dict[str, dict[str, float]] = {}
    for condition in sorted(set(condition_of.tolist())):
        index = np.nonzero(condition_of == condition)[0]
        row = []
        for level in TEST_LEVELS:
            frames = stacked(channels, level, MODALITIES)[test][index]
            reference = targets[test][index]
            row.append(float(np.mean(per_frame_iou(baseline, frames, reference,
                                                   gated=False))))
        iou_by_condition[condition] = {str(level): value
                                       for level, value in zip(TEST_LEVELS, row)}
        print(f"{condition:<12}{index.size:>7}"
              + "".join(f"{value:>9.3f}" for value in row))

    # The A2 ablation measured this baseline, so its s=0 value is known: agreeing
    # with it is what says the two studies share a frame set rather than merely
    # claiming to.
    a2_error_at_zero = 0.0037
    print(f"\nreproduction check: the A2 log reports 1-IoU={a2_error_at_zero:.4f} at s=0 for the "
          f"concatenated baseline\n  i.e. IoU={1 - a2_error_at_zero:.4f}; this run gives "
          f"{results['baseline (concat)'][0]:.4f}")

    target = 0.62
    best = results[gated_key][-1]
    print(f"\ntarget: s=4 IoU {results['baseline (concat)'][-1]:.3f} -> {target:.3f}; "
          f"gated gives {best:.3f}  [{'MET' if best >= target else 'not met yet'}]")

    # The result is an artifact, not just console output: runs are cheap to repeat
    # and two of them writing to one transcript is a mess, so each run writes its
    # own timestamped JSON under the versioned manifest directory.
    result = {
        "args": vars(args),
        "device": device,
        "frames": {"total": total, "train": train_count, "test": total - train_count},
        "iou": results,
        "delta_gated_minus_baseline": deltas,
        "reliability_table": table,
        "mean_gates": {str(level): mean_gates(gated, stacked(channels, level, MODALITIES)[test])
                       for level in TEST_LEVELS},
        "mean_gates_oracle": {str(level): mean_gates(gated, stacked(channels, level, MODALITIES)[test],
                                                     severity_override=level)
                              for level in TEST_LEVELS},
        "severity_path_shift": severity_shift,
        "iou_by_condition": iou_by_condition,
        "parameters": {"baseline": parameter_count(baseline), "gated": parameter_count(gated)},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    destination = MANIFEST_DIR / f"fusion_study_{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"result written to {destination}")
    print(f"elapsed: {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())