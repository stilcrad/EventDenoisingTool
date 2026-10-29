#!/usr/bin/env python3
"""Frozen gated dual-expert preparation and fixed-ROI EBOS evaluation.

The script creates new HDF5 streams and derived solver configurations under a
new run directory.  It reads the original scene configuration unchanged except
for the new event-HDF5 override and output directory.  The EBOS solver is
*opt-in* (``--run-solver``), so preparing this script never starts a long
multi-scene experiment by accident.

Every flow metric is computed only inside the original configuration ROI.  A
missing prediction frame, non-finite prediction, or unmatched timestamp counts
against coverage and failure.  The clean-stream EBOS result is a source-derived
reference, not an absolute physical flow ground truth.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml

from downstream_ebiv_gated import infer_gated_from_checkpoint, sha256_file


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK_ROOT = ROOT
DEFAULT_REPO = WORK_ROOT / "AMS2026 (2)" / "AMS_template_2022" / "_codex_event_based_bos_extract" / "event_based_bos-main"
DEFAULT_PACKAGE = WORK_ROOT / "AMS2026 (2)" / "experiments" / "ebos_experiments" / "experiment" / "ebos_multiscene_best"
DEFAULT_OUT = WORK_ROOT / "revision_eval" / "ai_module" / "runs" / "gated_v1" / "ebos"
DEFAULT_PYTHON = Path(sys.executable)
ABS_ERROR_PX = 0.5
REL_ERROR = 0.2


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )


def _absolute_path(path: Path) -> Path:
    """Make a path absolute without resolving the DenoisingASCII junction.

    OpenCV in the original EBOS stack cannot reliably open Unicode Windows
    paths.  ``Path.resolve()`` follows the ASCII junction back to the Chinese
    workspace path, so it must not be used for paths handed to ``bos_event``.
    """
    return Path(path).expanduser().absolute()


def _save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, default=_json_default)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _checkpoint_tag(path: Path) -> str:
    return f"{path.stem}_{sha256_file(path)[:10]}"


def _load_events(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"event HDF5 does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if "raw_events" not in handle:
            raise KeyError(f"{path} lacks raw_events")
        group = handle["raw_events"]
        missing = [key for key in ("x", "y", "t", "p") if key not in group]
        if missing:
            raise KeyError(f"{path} raw_events lacks {missing}")
        events = {key: np.asarray(group[key]) for key in ("x", "y", "t", "p")}
    lengths = {len(value) for value in events.values()}
    if len(lengths) != 1:
        raise ValueError(f"event arrays in {path} have inconsistent lengths")
    if len(events["t"]) and np.any(np.diff(events["t"].astype(np.int64)) < 0):
        raise ValueError(f"events in {path} are not chronologically sorted")
    return events


def _write_events(path: Path, events: Mapping[str, np.ndarray]) -> None:
    """Write a new HDF5 event stream without touching any historical HDF5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("raw_events")
        group.create_dataset("x", data=np.asarray(events["x"], dtype=np.uint16), compression="gzip", compression_opts=4)
        group.create_dataset("y", data=np.asarray(events["y"], dtype=np.uint16), compression="gzip", compression_opts=4)
        group.create_dataset("t", data=np.asarray(events["t"], dtype=np.int64), compression="gzip", compression_opts=4)
        group.create_dataset("p", data=np.asarray(events["p"], dtype=np.int16), compression="gzip", compression_opts=4)


def _event_keys(events: Mapping[str, np.ndarray], height: int, width: int) -> np.ndarray:
    """Losslessly encode t/x/y/p to match raw records to source-event copies."""
    t = np.asarray(events["t"], dtype=np.uint64)
    x = np.asarray(events["x"], dtype=np.uint64)
    y = np.asarray(events["y"], dtype=np.uint64)
    p = (np.asarray(events["p"]) > 0).astype(np.uint64)
    if np.any(x >= width) or np.any(y >= height):
        raise ValueError("events lie outside declared sensor dimensions")
    return (((t * np.uint64(width) + x) * np.uint64(height) + y) * np.uint64(2) + p)


def labels_from_clean(raw: Mapping[str, np.ndarray], clean: Mapping[str, np.ndarray], height: int, width: int) -> np.ndarray:
    """Recover source/injected proxy labels by exact multiset matching.

    For duplicate event tuples, the stable order retains as many raw records as
    occur in the source stream.  This reproduces the deterministic source-plus-
    noise construction; it does not claim source events are certified clean.
    """
    raw_keys = _event_keys(raw, height, width)
    clean_keys = np.sort(_event_keys(clean, height, width), kind="stable")
    order = np.argsort(raw_keys, kind="stable")
    sorted_raw = raw_keys[order]
    if not len(sorted_raw):
        return np.empty(0, dtype=bool)
    group_starts = np.r_[0, np.flatnonzero(sorted_raw[1:] != sorted_raw[:-1]) + 1]
    group_counts = np.diff(np.r_[group_starts, len(sorted_raw)])
    unique_clean, clean_counts = np.unique(clean_keys, return_counts=True)
    positions = np.searchsorted(unique_clean, sorted_raw[group_starts])
    matches = (positions < len(unique_clean)) & (
        unique_clean[np.minimum(positions, max(len(unique_clean) - 1, 0))] == sorted_raw[group_starts]
    ) if len(unique_clean) else np.zeros(len(group_starts), dtype=bool)
    source_counts = np.zeros(len(group_starts), dtype=np.int64)
    if len(unique_clean):
        source_counts[matches] = clean_counts[positions[matches]]
    starts_per_item = np.repeat(group_starts, group_counts)
    source_counts_per_item = np.repeat(source_counts, group_counts)
    source_sorted = (np.arange(len(sorted_raw)) - starts_per_item) < source_counts_per_item
    labels = np.empty(len(sorted_raw), dtype=bool)
    labels[order] = source_sorted
    return labels


def roi_mask(events: Mapping[str, np.ndarray], roi: Mapping[str, int]) -> np.ndarray:
    """Use EBOS's row/column convention: xmin/xmax index y, ymin/ymax index x."""
    return (
        (np.asarray(events["y"]) >= int(roi["xmin"]))
        & (np.asarray(events["y"]) < int(roi["xmax"]))
        & (np.asarray(events["x"]) >= int(roi["ymin"]))
        & (np.asarray(events["x"]) < int(roi["ymax"]))
    )


def _config_time_window(config: Mapping[str, Any]) -> Tuple[float, float]:
    windows = config.get("evaluation", {}).get("time_list", [])
    if not isinstance(windows, list) or len(windows) != 1 or not isinstance(windows[0], (list, tuple)) or len(windows[0]) != 2:
        raise ValueError("expected exactly one original evaluation.time_list window")
    start, end = float(windows[0][0]), float(windows[0][1])
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("invalid evaluation time window")
    return start, end


def _read_original_config(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, int], int, int, Tuple[float, float]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"original EBOS configuration not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"EBOS configuration must be a mapping: {config_path}")
    common = config.get("common_params")
    data = config.get("data")
    if not isinstance(common, Mapping) or not isinstance(data, Mapping):
        raise KeyError("EBOS configuration lacks common_params or data")
    roi = {key: int(common[key]) for key in ("xmin", "xmax", "ymin", "ymax")}
    height, width = int(data["height"]), int(data["width"])
    if not (0 <= roi["xmin"] < roi["xmax"] <= height and 0 <= roi["ymin"] < roi["ymax"] <= width):
        raise ValueError(f"invalid fixed ROI {roi} for {height}x{width}")
    return config, roi, height, width, _config_time_window(config)


def _resolve_config_output(config_path: Path, config: Mapping[str, Any], repo: Path) -> Path:
    output = Path(str(config.get("output_dir", "")))
    if not str(output):
        raise KeyError("original EBOS config lacks output_dir")
    # The original configs are evaluated with event_based_bos-main as CWD.
    return _absolute_path(output if output.is_absolute() else repo / output)


def _write_derived_config(original: Mapping[str, Any], hdf5_path: Path, output_dir: Path, destination: Path) -> str:
    """Preserve original settings while replacing only input HDF5 and output path."""
    config = json.loads(json.dumps(original))  # Plain deep copy avoids modifying caller state.
    config["data"]["event_hdf_override"] = str(_absolute_path(hdf5_path))
    config["output_dir"] = str(_absolute_path(output_dir))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return sha256_file(destination)


def flow_files(directory: Path) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    if not directory.is_dir():
        return result
    for path in directory.glob("pred_flow*.npy"):
        match = re.fullmatch(r"pred_flow(\d+)\.npy", path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def timestamps(directory: Path) -> Dict[int, Tuple[float, float]]:
    result: Dict[int, Tuple[float, float]] = {}
    path = directory / "timestamps_per_frame.txt"
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"frame (\d+)::(.*)", line.strip())
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group(2))
            result[int(match.group(1))] = (float(value["t1"]), float(value["t2"]))
        except (ValueError, SyntaxError, KeyError, TypeError):
            continue
    return result


def stable_array(path: Path) -> np.ndarray:
    before = path.stat()
    array = np.load(path, allow_pickle=False)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"prediction changed while it was read: {path}")
    return np.asarray(array, dtype=np.float64)


def crop_to_roi(flow: np.ndarray, roi: Mapping[str, int], expected_shape: Tuple[int, int, int]) -> np.ndarray:
    if flow.shape != expected_shape:
        raise ValueError(f"flow shape {flow.shape} differs from expected {expected_shape}")
    return flow[:, int(roi["xmin"]):int(roi["xmax"]), int(roi["ymin"]):int(roi["ymax"])]


def fixed_roi_flow_metrics(
    *,
    scene: str,
    split: str,
    method: str,
    seed: Any,
    reference_dir: Path,
    prediction_dir: Path,
    roi: Mapping[str, int],
    expected_shape: Tuple[int, int, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Match frames by timestamps and score only the fixed original ROI.

    All reference frames contribute their finite ROI pixels to the denominator.
    A missing or unusable prediction frame supplies zero available pixels and
    therefore worsens coverage/failure instead of being removed by ``zip``.
    """
    references, predictions = flow_files(reference_dir), flow_files(prediction_dir)
    reference_times, prediction_times = timestamps(reference_dir), timestamps(prediction_dir)
    reference_ids = sorted(set(references) | set(reference_times))
    prediction_by_time: Dict[Tuple[int, int], int] = {}
    duplicate_times: set = set()
    for index, pair in prediction_times.items():
        key = tuple(int(round(value * 1e6)) for value in pair)
        if key in prediction_by_time:
            duplicate_times.add(key)
        prediction_by_time[key] = index

    frames: List[Dict[str, Any]] = []
    errors: List[np.ndarray] = []
    reference_pixels = available_pixels = missing_pixels = error_failures = 0
    used_prediction_indices: set = set()
    for index in reference_ids:
        pair = reference_times.get(index)
        key = tuple(int(round(value * 1e6)) for value in pair) if pair else None
        prediction_index = prediction_by_time.get(key) if key is not None and key not in duplicate_times else None
        prediction_path = predictions.get(prediction_index) if prediction_index is not None else None
        row: Dict[str, Any] = {
            "scene": scene,
            "split": split,
            "method": method,
            "seed": seed,
            "reference_frame": index,
            "prediction_frame": prediction_index,
            "t1_s": pair[0] if pair else None,
            "t2_s": pair[1] if pair else None,
            "status": "missing_prediction",
            "detail": "",
            "reference_path": str(references.get(index)) if index in references else None,
            "prediction_path": str(prediction_path) if prediction_path else None,
            "reference_valid_pixels": 0,
            "prediction_available_pixels": 0,
            "missing_pixels": 0,
            "error_failure_pixels": 0,
            "coverage": 0.0,
            "failure": 1.0,
            "epe_mean_px": None,
            "epe_median_px": None,
            "epe_p90_px": None,
        }
        try:
            if index not in references or pair is None:
                raise ValueError("reference flow or timestamp is missing")
            reference = crop_to_roi(stable_array(references[index]), roi, expected_shape)
            valid_reference = np.isfinite(reference).all(axis=0)
            n_reference = int(valid_reference.sum())
            if n_reference == 0:
                raise ValueError("reference ROI has no finite vector")
        except (ValueError, OSError, EOFError) as exc:
            row.update(status="invalid_reference", detail=str(exc))
            frames.append(row)
            continue

        reference_pixels += n_reference
        row["reference_valid_pixels"] = n_reference
        row["missing_pixels"] = n_reference
        if prediction_path is not None:
            try:
                prediction = crop_to_roi(stable_array(prediction_path), roi, expected_shape)
                available = valid_reference & np.isfinite(prediction).all(axis=0)
                error = np.hypot(prediction[0] - reference[0], prediction[1] - reference[1])
                limit = np.maximum(ABS_ERROR_PX, REL_ERROR * np.hypot(reference[0], reference[1]))
                error_values = error[available]
                n_available = int(available.sum())
                n_bad = int((available & (error > limit)).sum())
                row.update(
                    status="complete",
                    prediction_available_pixels=n_available,
                    missing_pixels=n_reference - n_available,
                    error_failure_pixels=n_bad,
                    coverage=n_available / n_reference,
                    failure=(n_reference - n_available + n_bad) / n_reference,
                    epe_mean_px=float(error_values.mean()) if error_values.size else None,
                    epe_median_px=float(np.median(error_values)) if error_values.size else None,
                    epe_p90_px=float(np.quantile(error_values, 0.90)) if error_values.size else None,
                )
                if error_values.size:
                    errors.append(error_values)
                used_prediction_indices.add(prediction_index)
            except (ValueError, OSError, EOFError) as exc:
                row.update(status="invalid_prediction", detail=str(exc))
        available_pixels += int(row["prediction_available_pixels"])
        missing_pixels += int(row["missing_pixels"])
        error_failures += int(row["error_failure_pixels"])
        frames.append(row)

    all_errors = np.concatenate(errors) if errors else np.empty(0, dtype=float)
    completed = sum(row["status"] == "complete" for row in frames)
    unmatched = sorted(set(predictions) - used_prediction_indices)
    status = "complete" if reference_ids and completed == len(reference_ids) and not unmatched else ("partial" if predictions else "pending")
    if not reference_ids or any(row["status"] == "invalid_reference" for row in frames):
        status = "invalid_reference"
    summary = {
        "scene": scene,
        "split": split,
        "method": method,
        "seed": seed,
        "status": status,
        "reference_frames": len(reference_ids),
        "completed_frames": completed,
        "missing_or_invalid_frames": len(reference_ids) - completed,
        "unmatched_prediction_frames": unmatched,
        "roi": dict(roi),
        "reference_valid_pixels": reference_pixels,
        "prediction_available_pixels": available_pixels,
        "missing_pixels": missing_pixels,
        "error_failure_pixels": error_failures,
        "coverage": available_pixels / reference_pixels if reference_pixels else 0.0,
        "failure": (missing_pixels + error_failures) / reference_pixels if reference_pixels else None,
        "epe_mean_px": float(all_errors.mean()) if all_errors.size else None,
        "epe_median_px": float(np.median(all_errors)) if all_errors.size else None,
        "epe_p90_px": float(np.quantile(all_errors, 0.90)) if all_errors.size else None,
        "reference_dir": str(reference_dir),
        "prediction_dir": str(prediction_dir),
        "reference_timestamps_sha256": sha256_file(reference_dir / "timestamps_per_frame.txt") if (reference_dir / "timestamps_per_frame.txt").is_file() else None,
        "prediction_timestamps_sha256": sha256_file(prediction_dir / "timestamps_per_frame.txt") if (prediction_dir / "timestamps_per_frame.txt").is_file() else None,
    }
    return frames, summary


def _run_solver(
    *,
    python: Path,
    repo: Path,
    config_path: Path,
    output_dir: Path,
    event_hash: str,
    config_hash: str,
    run_solver: bool,
    log_path: Path,
) -> Tuple[float, bool]:
    """Run the untouched EBOS solver only after a hash-aware cache check."""
    provenance_path = output_dir / "gated_solver_provenance.json"
    expected = {"event_hdf5_sha256": event_hash, "config_sha256": config_hash, "solver": "bos_event.py --eval"}
    if (output_dir / "pred_flow0.npy").is_file() and provenance_path.is_file():
        try:
            prior = json.loads(provenance_path.read_text(encoding="utf-8"))
            if all(prior.get(key) == value for key, value in expected.items()):
                return 0.0, True
        except (OSError, ValueError):
            pass
    if not run_solver:
        return 0.0, False
    if not python.is_file():
        raise FileNotFoundError(f"EBOS python environment not found: {python}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(python), str(repo / "bos_event.py"), "--config_file", str(config_path), "--eval"],
            cwd=str(repo),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(f"EBOS solver failed for {config_path}; inspect {log_path}")
    _dump_json(provenance_path, {**expected, "solver_wall_s": elapsed, "config": str(config_path)})
    return elapsed, False


def _split_for_scene(scene: str) -> str:
    protocol = HERE / "protocol.json"
    if not protocol.is_file():
        return "legacy"
    data = json.loads(protocol.read_text(encoding="utf-8"))
    for split, scenes in data.get("splits", {}).items():
        if scene in scenes:
            return str(split)
    return "legacy"


def _scene_paths(args: argparse.Namespace, scene: str) -> Tuple[Path, Path, Path, Optional[Path]]:
    config_path = _absolute_path(args.config) if args.config else _absolute_path(args.repo / "configs" / f"ebos_ms_{scene}_clean.yaml")
    raw_hdf = _absolute_path(args.raw_hdf) if args.raw_hdf else _absolute_path(args.package / "hdf5" / scene / "raw_noise20.hdf5")
    clean_hdf = _absolute_path(args.clean_hdf) if args.clean_hdf else _absolute_path(args.package / "hdf5" / scene / "clean_events.hdf5")
    return config_path, raw_hdf, clean_hdf, _absolute_path(args.reference_dir) if args.reference_dir else None


def run_scene(args: argparse.Namespace, scene: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    config_path, raw_hdf_path, clean_hdf_path, explicit_reference = _scene_paths(args, scene)
    original_config, roi, height, width, window_s = _read_original_config(config_path)
    eval_start_us, eval_end_us = int(round(window_s[0] * 1e6)), int(round(window_s[1] * 1e6))
    raw, clean = _load_events(raw_hdf_path), _load_events(clean_hdf_path)
    labels = labels_from_clean(raw, clean, height, width)
    raw_array = np.column_stack([raw[key] for key in ("t", "x", "y", "p")]).astype(np.int64)
    raw_array[:, 3] = (raw_array[:, 3] > 0).astype(np.int64)
    raw_roi = roi_mask(raw, roi)
    roi_indices = np.flatnonzero(raw_roi)
    roi_events = raw_array[roi_indices]
    if not len(roi_events):
        raise ValueError(f"{scene}: original ROI contains no events")
    local_eval = (roi_events[:, 0] >= eval_start_us) & (roi_events[:, 0] < eval_end_us)
    if not int(local_eval.sum()):
        raise ValueError(f"{scene}: original evaluation time window has no ROI events")
    labels_eval = labels[roi_indices][local_eval]
    reference_dir = explicit_reference if explicit_reference is not None else _resolve_config_output(config_path, original_config, args.repo)
    historical_raw = reference_dir.with_name(reference_dir.name.replace("_clean", "_raw"))
    historical_vgs = reference_dir.with_name(reference_dir.name.replace("_clean", "_ours"))
    if args.run_solver and not reference_dir.is_dir():
        raise FileNotFoundError(
            f"{scene}: source-reference EBOS output is absent: {reference_dir}. "
            "Run or locate the untouched original clean solver output before scoring a new denoiser."
        )

    config_hash, raw_hash, clean_hash = sha256_file(config_path), sha256_file(raw_hdf_path), sha256_file(clean_hdf_path)
    event_rows: List[Dict[str, Any]] = []
    flow_rows: List[Dict[str, Any]] = []
    scene_metadata: Dict[str, Any] = {
        "scene": scene,
        "split": _split_for_scene(scene),
        "original_config": str(config_path),
        "original_config_sha256": config_hash,
        "raw_hdf5": str(raw_hdf_path),
        "raw_hdf5_sha256": raw_hash,
        "clean_hdf5": str(clean_hdf_path),
        "clean_hdf5_sha256": clean_hash,
        "reference_dir": str(reference_dir),
        "roi": roi,
        "evaluation_time_s": list(window_s),
        "selection_state": str(args.selection_state),
        "evaluation_events": int(local_eval.sum()),
        "evaluation_source_proxy_events": int(labels_eval.sum()),
        "evaluation_injected_proxy_events": int((~labels_eval).sum()),
        "rows": [],
    }

    # Read-only historical baselines receive the same fixed-ROI audit.  They
    # are never generated, replaced, or used to tune a new checkpoint.
    expected_shape = (2, height, width)
    for method, directory in (("Raw (historical)", historical_raw), ("VGS (historical)", historical_vgs)):
        frames, summary = fixed_roi_flow_metrics(
            scene=scene,
            split=_split_for_scene(scene),
            method=method,
            seed="",
            reference_dir=reference_dir,
            prediction_dir=directory,
            roi=roi,
            expected_shape=expected_shape,
        )
        flow_rows.extend(frames)
        scene_metadata["rows"].append(summary)

    for checkpoint_path in args.checkpoint:
        checkpoint_path = _absolute_path(checkpoint_path)
        tag = _checkpoint_tag(checkpoint_path)
        started = time.perf_counter()
        inference = infer_gated_from_checkpoint(
            roi_events,
            height,
            width,
            checkpoint_path,
            batch_size=args.batch_size,
            return_features=False,
        )
        inference_s = time.perf_counter() - started
        keep_roi = np.asarray(inference["keep"], dtype=bool)
        keep_full = np.ones(len(raw_array), dtype=bool)
        keep_full[roi_indices] = keep_roi
        gated_events = {key: np.asarray(raw[key])[keep_full] for key in ("x", "y", "t", "p")}
        learned_prefix = str(args.learned_method_prefix)
        learned_label = str(args.learned_method_label)
        learned_kind = str(args.learned_method_kind)
        hdf_path = args.out_dir / "hdf5" / scene / f"{learned_prefix}_{tag}.hdf5"
        _write_events(hdf_path, gated_events)
        hdf_hash = sha256_file(hdf_path)
        solver_dir = args.out_dir / "solver" / scene / f"{learned_prefix}_{tag}"
        derived_config = args.out_dir / "configs" / scene / f"{learned_prefix}_{tag}.yaml"
        derived_config_hash = _write_derived_config(original_config, hdf_path, solver_dir, derived_config)
        solver_s, solver_reused = _run_solver(
            python=args.python,
            repo=args.repo,
            config_path=derived_config,
            output_dir=solver_dir,
            event_hash=hdf_hash,
            config_hash=derived_config_hash,
            run_solver=args.run_solver,
            log_path=args.out_dir / "logs" / scene / f"{learned_prefix}_{tag}.log",
        )
        keep_eval = keep_roi[local_eval]
        source = labels_eval
        noise = ~source
        true_positive = int((keep_eval & source).sum())
        false_positive = int((keep_eval & noise).sum())
        alpha_value = inference.get("alpha")
        alpha = (np.asarray(alpha_value)[local_eval] if alpha_value is not None
                 else np.empty(0, dtype=np.float32))
        physics_value = inference.get("p_physics")
        ai_value = inference.get("p_ai")
        p_physics = (np.asarray(physics_value)[local_eval] if physics_value is not None
                     else np.empty(0, dtype=np.float32))
        p_ai = (np.asarray(ai_value)[local_eval] if ai_value is not None
                else np.empty(0, dtype=np.float32))
        event_row = {
            "scene": scene,
            "split": _split_for_scene(scene),
            "method": learned_label,
            "method_kind": learned_kind,
            "training_seed": inference.get("training_seed"),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": inference["checkpoint_sha256"],
            "checkpoint_schema": inference["checkpoint_schema"],
            "threshold": inference["threshold"],
            "raw_events_total": int(len(raw_array)),
            "roi_events_scored": int(len(roi_events)),
            "roi_events_retained": int(keep_roi.sum()),
            "evaluation_events": int(local_eval.sum()),
            "evaluation_source_recall": true_positive / max(int(source.sum()), 1),
            "evaluation_noise_rejection": 1.0 - false_positive / max(int(noise.sum()), 1),
            "evaluation_retention": float(keep_eval.mean()),
            "evaluation_precision": true_positive / max(true_positive + false_positive, 1),
            "mean_alpha": float(alpha.mean()) if len(alpha) else None,
            "std_alpha": float(alpha.std()) if len(alpha) else None,
            "mean_expert_disagreement": float(np.abs(p_physics - p_ai).mean()) if len(p_physics) and len(p_ai) else None,
            "learned_inference_metadata": inference.get("metadata"),
            "feature_and_inference_wall_s": inference_s,
            "solver_wall_s": solver_s,
            "solver_reused": solver_reused,
            "generated_hdf5": str(hdf_path),
            "generated_hdf5_sha256": hdf_hash,
            "derived_config": str(derived_config),
            "derived_config_sha256": derived_config_hash,
        }
        event_rows.append(event_row)
        frames, summary = fixed_roi_flow_metrics(
            scene=scene,
            split=_split_for_scene(scene),
            method=learned_label,
            seed=inference.get("training_seed", checkpoint_path.stem),
            reference_dir=reference_dir,
            prediction_dir=solver_dir,
            roi=roi,
            expected_shape=expected_shape,
        )
        flow_rows.extend(frames)
        summary.update(
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=inference["checkpoint_sha256"],
            threshold=inference["threshold"],
            solver_wall_s=solver_s,
            solver_reused=solver_reused,
            hdf5_path=str(hdf_path),
            hdf5_sha256=hdf_hash,
            derived_config=str(derived_config),
            derived_config_sha256=derived_config_hash,
        )
        scene_metadata["rows"].append(summary)
    return event_rows, flow_rows, scene_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare frozen gated event streams and fixed-ROI EBOS metrics; solver execution is opt-in."
    )
    parser.add_argument("--scene", required=True, nargs="+", help="one or more original EBOS scene names")
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+", help="one or three frozen gated checkpoints")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="event_based_bos-main repository")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE, help="historical EBOS multiscene package")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="new gated result directory")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable for the original EBOS solver")
    parser.add_argument("--config", type=Path, default=None, help="override original config (only for one --scene)")
    parser.add_argument("--raw-hdf", type=Path, default=None, help="override raw noisy HDF5 (only for one --scene)")
    parser.add_argument("--clean-hdf", type=Path, default=None, help="override clean/source HDF5 for proxy-label matching (only for one --scene)")
    parser.add_argument("--reference-dir", type=Path, default=None, help="override clean-source solver-output directory")
    parser.add_argument("--batch-size", type=int, default=65536, help="CPU gated inference batch size")
    parser.add_argument("--run-solver", action="store_true", help="run original bos_event.py; absent by default to prevent accidental long runs")
    parser.add_argument("--learned-method-prefix", default="gated", help="filename prefix for learned outputs")
    parser.add_argument("--learned-method-label", default="Gated dual-expert (frozen)", help="human-readable learned method label")
    parser.add_argument("--learned-method-kind", default="gated_dual_expert", help="semantic learned method identifier")
    parser.add_argument("--result-stem", default="ebos_gated", help="basename for CSV and metadata outputs")
    parser.add_argument("--protocol-name", default="frozen_gated_dual_expert_on_original_ebos_config_roi_and_time_window")
    parser.add_argument("--selection-state", default="all raw chronological events within the fixed ROI; selection does not feed back into causal state")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.repo = _absolute_path(args.repo)
    args.package = _absolute_path(args.package)
    args.out_dir = _absolute_path(args.out_dir)
    args.python = _absolute_path(args.python)
    if len(args.scene) > 1 and any(value is not None for value in (args.config, args.raw_hdf, args.clean_hdf, args.reference_dir)):
        raise ValueError("--config/--raw-hdf/--clean-hdf/--reference-dir require exactly one --scene")
    if not args.repo.is_dir():
        raise FileNotFoundError(f"event_based_bos-main repository not found: {args.repo}")
    if not args.package.is_dir() and any(value is None for value in (args.raw_hdf, args.clean_hdf)):
        raise FileNotFoundError(f"historical EBOS package not found: {args.package}")
    if len(set(args.checkpoint)) != len(args.checkpoint):
        raise ValueError("each frozen checkpoint must appear once")
    event_rows: List[Dict[str, Any]] = []
    flow_rows: List[Dict[str, Any]] = []
    scene_metadata: List[Dict[str, Any]] = []
    for scene in args.scene:
        ev, flows, metadata = run_scene(args, scene)
        event_rows.extend(ev)
        flow_rows.extend(flows)
        scene_metadata.append(metadata)
        print(f"prepared {scene}", flush=True)
    _save_csv(args.out_dir / f"{args.result_stem}_event_metrics.csv", event_rows)
    _save_csv(args.out_dir / f"{args.result_stem}_flow_frames.csv", flow_rows)
    summary_rows = [row for scene in scene_metadata for row in scene["rows"]]
    _save_csv(args.out_dir / f"{args.result_stem}_flow_summary.csv", summary_rows)
    metadata = {
        "protocol": str(args.protocol_name),
        "status": "completed" if args.run_solver else "prepared_no_solver",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": str(args.repo),
        "package": str(args.package),
        "out_dir": str(args.out_dir),
        "checkpoints": [{"path": str(_absolute_path(path)), "sha256": sha256_file(path)} for path in args.checkpoint],
        "reference": "clean-source EBOS solver output; source-derived reference, not absolute physical ground truth",
        "metrics": {
            "roi": "only original common_params ROI; no full-frame zero-padding dilution",
            "matching": "t1/t2 timestamps at microsecond precision",
            "coverage": "available finite prediction ROI pixels / finite reference ROI pixels across all reference frames",
            "failure": f"missing/non-finite prediction OR EPE > max({ABS_ERROR_PX} px, {REL_ERROR} * reference displacement)",
            "epe": "available finite ROI pixel displacement error in px per image-pair interval",
        },
        "scenes": scene_metadata,
    }
    _dump_json(args.out_dir / f"{args.result_stem}_metadata.json", metadata)
    print(json.dumps({"status": metadata["status"], "out_dir": str(args.out_dir), "scenes": len(args.scene)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
