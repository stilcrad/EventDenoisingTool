#!/usr/bin/env python3
"""Frozen gated dual-expert inference on the original EBIV benchmark.

This is a *new* benchmark wrapper.  It reads the original EBIV event file and
solver, recreates the historical uniform-noise RNG protocol, and writes only
new artifacts beneath ``--out-dir``.  It never edits the historical VGS,
Raw, or manuscript outputs.

The gated checkpoint must contain a frozen ``GatedDualExpert`` state dict,
training-set generic-feature normalization, and a validation-selected event
threshold.  See ``README.md`` in this directory for the compact schema.

Example after the trainer has produced three frozen checkpoints::

    python downstream_ebiv_gated.py --checkpoint runs/gated_v1/models/gated_4101.pt \
        runs/gated_v1/models/gated_4102.pt runs/gated_v1/models/gated_4103.pt \
        --run-solver

The EBIV source-event velocity map is used only as a source-derived reference;
it is not an independent physical ground truth.  Missing vectors count against
coverage and failure rather than disappearing from the error calculation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EBASE = ROOT / "AMS2026 (2)" / "AMS_template_2022"
DEFAULT_EVT = EBASE / "ebiv-main" / "sample_data" / "wallflow4_dense_3_100ms.evt"
DEFAULT_CFG = EBASE / "ebiv-main" / "sample_data" / "ebiv_params.cfg"
DEFAULT_OUT = HERE / "runs" / "gated_v1" / "ebiv"

# Imported lazily in ``_ebiv_module`` so that ``--help`` and checkpoint-only
# inference remain usable before an EBIV environment is configured.


def sha256_file(path: Union[str, Path]) -> str:
    """Return a stable SHA-256 hash of a file without loading it all at once."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash array values and layout after forcing a contiguous representation."""
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


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


def _ebiv_module():
    """Load the untouched original EBIV module only when a solver is needed."""
    for candidate in (HERE, EBASE):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    import ebiv_noise_vector_experiment as original  # noqa: WPS433

    return original


def _load_checkpoint(checkpoint: Union[str, Path, Mapping[str, Any]]) -> Tuple[Dict[str, Any], Optional[Path]]:
    if isinstance(checkpoint, Mapping):
        return dict(checkpoint), None
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"gated checkpoint not found: {path}")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the ``weights_only`` argument.
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError(f"checkpoint {path} must deserialize to a mapping")
    return dict(loaded), path


def _first_mapping_value(mapping: Mapping[str, Any], keys: Iterable[str]) -> Tuple[Any, Optional[str]]:
    for key in keys:
        if key in mapping:
            return mapping[key], key
    return None, None


def _checkpoint_normalization(
    checkpoint: Mapping[str, Any], *, branch: str, width: int, allow_identity: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover exactly the training normalization for one checkpoint branch."""
    if branch == "generic":
        mean_keys = ("feature_mean", "generic_feature_mean", "mean")
        scale_keys = ("feature_scale", "generic_feature_scale", "scale", "std")
        nested = checkpoint.get("generic_normalization") or checkpoint.get("normalization")
    elif branch == "physics":
        mean_keys = ("physics_feature_mean", "physics_mean")
        scale_keys = ("physics_feature_scale", "physics_scale", "physics_std")
        nested = checkpoint.get("physics_normalization")
    else:
        raise ValueError(f"unknown normalization branch: {branch}")
    mean, mean_key = _first_mapping_value(checkpoint, mean_keys)
    scale, scale_key = _first_mapping_value(checkpoint, scale_keys)
    normalization = nested
    if isinstance(normalization, Mapping):
        if mean is None:
            mean, mean_key = _first_mapping_value(normalization, ("mean", "feature_mean"))
        if scale is None:
            scale, scale_key = _first_mapping_value(normalization, ("scale", "std", "feature_scale"))
    if mean is None or scale is None:
        if allow_identity:
            return np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
        raise KeyError(
            f"checkpoint lacks {width}-D {branch} normalization"
        )
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    scale_array = np.asarray(scale, dtype=np.float32).reshape(-1)
    if mean_array.shape != (width,) or scale_array.shape != (width,):
        raise ValueError(
            f"{branch} normalization must have {width} values, got mean {mean_array.shape} ({mean_key}) "
            f"and scale {scale_array.shape} ({scale_key})"
        )
    if not np.isfinite(mean_array).all() or not np.isfinite(scale_array).all() or np.any(scale_array <= 0):
        raise ValueError(f"checkpoint {branch} normalization must be finite with positive scales")
    return mean_array, scale_array


def _checkpoint_threshold(checkpoint: Mapping[str, Any], threshold: Optional[float]) -> float:
    value = threshold
    if value is None:
        value, _ = _first_mapping_value(checkpoint, ("threshold", "selection_threshold", "event_threshold"))
    if value is None:
        all_thresholds = checkpoint.get("thresholds")
        if isinstance(all_thresholds, Mapping):
            gated = all_thresholds.get("gated_dual_expert", all_thresholds.get("gated"))
            if isinstance(gated, Mapping) and gated:
                # The trainer stores e.g. {"0.98": threshold, "0.95": ...}.
                # Choose the most conservative source-recall operating point.
                candidates = []
                for recall, candidate in gated.items():
                    try:
                        candidates.append((float(recall), float(candidate)))
                    except (TypeError, ValueError):
                        continue
                if candidates:
                    value = max(candidates, key=lambda item: item[0])[1]
    if value is None:
        raise KeyError("no selection threshold supplied and checkpoint contains none")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"selection threshold must be in [0, 1], got {value}")
    return value


def _checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state, _ = _first_mapping_value(checkpoint, ("model_state_dict", "state_dict", "model"))
    if isinstance(state, Mapping):
        return state
    # ``train_gated_model.py`` intentionally serializes the three frozen
    # branches separately so its checkpoint can also hold ablation networks.
    branch_specs = (
        ("physics_calibrator", "physics_calibration_state"),
        ("ai_expert", "ai_expert_state"),
        ("gate", "gate_state"),
    )
    assembled: Dict[str, Any] = {}
    for prefix, key in branch_specs:
        branch = checkpoint.get(key)
        if not isinstance(branch, Mapping):
            raise KeyError(
                "checkpoint lacks model_state_dict and separate frozen gated branches "
                f"({key})"
            )
        assembled.update({f"{prefix}.{name}": value for name, value in branch.items()})
    return assembled


def _build_gated_model(checkpoint: Mapping[str, Any]):
    """Instantiate the CPU gated model and restore a small canonical checkpoint.

    A calibration dictionary can fill the two calibration weights when a
    serializer stored the branch separately.  All AI and gate weights remain
    mandatory, so a generic MLP checkpoint is never silently accepted.
    """
    schema = str(checkpoint.get("model_schema", checkpoint.get("schema", "vgs_physics_gated_dual_expert_v1")))
    if schema not in {
        "vgs_physics_gated_dual_expert_v1",
        "gated_dual_expert_v1",
        "gated-dual-expert-v1",
    }:
        raise ValueError(f"unsupported gated checkpoint schema: {schema!r}")
    from gated_model import AIExpert, GatedDualExpert, PhysicsCalibrator, ReliabilityGate  # noqa: WPS433

    calibration = checkpoint.get("physics_calibration")
    eps = float(calibration.get("eps", 1.0e-8)) if isinstance(calibration, Mapping) else 1.0e-8
    model = GatedDualExpert(
        physics_calibrator=PhysicsCalibrator(eps=eps),
        ai_expert=AIExpert(),
        gate=ReliabilityGate(),
    )
    state = dict(_checkpoint_state_dict(checkpoint))
    if isinstance(calibration, Mapping):
        for destination, source in (("physics_calibrator.a", "a"), ("physics_calibrator.b", "b")):
            if destination not in state and source in calibration:
                state[destination] = torch.tensor(float(calibration[source]), dtype=torch.float32)
    try:
        result = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        required = [
            "ai_expert.network.0.weight",
            "gate.network.0.weight",
            "physics_calibrator.a",
            "physics_calibrator.b",
        ]
        missing = [key for key in required if key not in state]
        detail = f" missing canonical keys: {missing}." if missing else ""
        raise ValueError(f"checkpoint is not a complete frozen GatedDualExpert.{detail}") from exc
    if result.missing_keys or result.unexpected_keys:  # Defensive: strict=True should already enforce this.
        raise ValueError(f"checkpoint key mismatch: {result}")
    model.cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def infer_gated_from_checkpoint(
    events: np.ndarray,
    height: int,
    width: int,
    checkpoint: Union[str, Path, Mapping[str, Any]],
    *,
    threshold: Optional[float] = None,
    batch_size: int = 65536,
    return_features: bool = False,
) -> Dict[str, Any]:
    """Causally score any chronological event stream with one frozen checkpoint.

    ``events`` is an ``N x 4`` integer array ``(t, x, y, p)``.  State is updated
    from every raw chronological input event by :func:`features.extract`; the
    returned mask only selects output events and never feeds back into state.
    Generic descriptors and, when their training checkpoint contains it,
    physics descriptors are standardized with their recorded training-only
    statistics.  The continuous VGS score itself is never normalized and goes
    to the frozen physics calibrator unchanged.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("height and width must be positive")
    data = np.asarray(events, dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("events must have shape (N, 4) in t,x,y,p order")
    checkpoint_data, checkpoint_path = _load_checkpoint(checkpoint)
    selected_threshold = _checkpoint_threshold(checkpoint_data, threshold)
    generic_mean, generic_scale = _checkpoint_normalization(
        checkpoint_data, branch="generic", width=6
    )
    # The canonical compact schema may deliberately pass raw physics features;
    # the trainer's separate-branch schema stores an 8-D normalization and is
    # reproduced exactly when present.
    physics_mean, physics_scale = _checkpoint_normalization(
        checkpoint_data, branch="physics", width=8, allow_identity=True
    )
    model = _build_gated_model(checkpoint_data)

    from features import extract  # noqa: WPS433

    features, scores = extract(data, int(height), int(width))
    generic_raw = np.asarray(features[:, :6], dtype=np.float32)
    physics_raw = np.asarray(features[:, 6:14], dtype=np.float32)
    generic = np.clip((generic_raw - generic_mean) / generic_scale, -20.0, 20.0).astype(np.float32, copy=False)
    physics = np.clip((physics_raw - physics_mean) / physics_scale, -20.0, 20.0).astype(np.float32, copy=False)
    score_column = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    outputs = {key: [] for key in ("p_physics", "p_ai", "alpha", "p_fused")}
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            component = model.components(generic[start:stop], physics[start:stop], score_column[start:stop])
            detached = component.detached_numpy()
            for key in outputs:
                outputs[key].append(detached[key].astype(np.float32, copy=False))
    probabilities = {
        key: np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        for key, parts in outputs.items()
    }
    result: Dict[str, Any] = {
        **probabilities,
        "keep": probabilities["p_fused"] >= selected_threshold,
        "threshold": selected_threshold,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "checkpoint_schema": str(checkpoint_data.get("model_schema", checkpoint_data.get("schema", "vgs_physics_gated_dual_expert_v1"))),
        "training_seed": checkpoint_data.get("seed", checkpoint_data.get("training_seed")),
        "feature_state": str(
            checkpoint_data.get("feature_state_rule", checkpoint_data.get("feature_state", "features.py causal raw chronological stream"))
        ),
    }
    if return_features:
        result.update(
            generic_raw=generic_raw,
            generic_normalized=generic,
            physics_features_raw=physics_raw,
            physics_features_normalized=physics,
            physics_scores=score_column.reshape(-1),
        )
    return result


def exact_labeled_noise(ev: Mapping[str, Any], rate: float, rng: np.random.Generator) -> Tuple[Dict[str, Any], np.ndarray]:
    """Recreate ``add_random_noise`` byte-for-byte and propagate proxy labels.

    The order of random draws is timestamps, x, y, then polarity.  The one
    generator is intentionally passed from rate to rate, matching the original
    EBIV script's legacy 10/20/40 percent protocol.
    """
    if rate < 0:
        raise ValueError("noise rate cannot be negative")
    height, width = [int(value) for value in ev["image_size"]]
    n_source = int(np.asarray(ev["t"]).size)
    n_noise = int(round(float(rate) * n_source))
    t_min, t_max = int(np.min(ev["t"])), int(np.max(ev["t"]))
    noisy = {
        "t": np.concatenate(
            [np.asarray(ev["t"], dtype=np.uint32), rng.integers(t_min, t_max + 1, n_noise, dtype=np.uint32)]
        ),
        "x": np.concatenate(
            [np.asarray(ev["x"], dtype=np.uint16), rng.integers(0, width, n_noise, dtype=np.uint16)]
        ),
        "y": np.concatenate(
            [np.asarray(ev["y"], dtype=np.uint16), rng.integers(0, height, n_noise, dtype=np.uint16)]
        ),
        "p": np.concatenate(
            [np.asarray(ev["p"], dtype=np.uint8), rng.integers(0, 2, n_noise, dtype=np.uint8)]
        ),
        "time_stamp": int(ev["time_stamp"]),
        "image_size": [height, width],
    }
    labels = np.concatenate([np.ones(n_source, dtype=np.uint8), np.zeros(n_noise, dtype=np.uint8)])
    order = np.argsort(noisy["t"], kind="mergesort")
    for key in ("t", "x", "y", "p"):
        noisy[key] = noisy[key][order]
    return noisy, labels[order]


def _decision_metrics(labels: np.ndarray, keep: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    if labels.shape != keep.shape:
        raise ValueError("labels and keep mask differ in shape")
    source = labels
    noise = ~labels
    true_positive = int((keep & source).sum())
    false_positive = int((keep & noise).sum())
    return {
        "source_recall": true_positive / max(int(source.sum()), 1),
        "noise_rejection": 1.0 - false_positive / max(int(noise.sum()), 1),
        "retention": float(keep.mean()) if len(keep) else float("nan"),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "n_source": int(source.sum()),
        "n_injected": int(noise.sum()),
    }


def coverage_metrics(original: Any, prediction: np.ndarray, reference: np.ndarray, reference_grid: np.ndarray) -> Dict[str, Any]:
    """Coverage-aware EBIV metrics on every valid source-reference vector."""
    field = original.vector_fields(prediction)
    ref = original.vector_fields(reference)
    if reference_grid.shape != field["vx"].shape:
        raise ValueError("reference grid and predicted EBIV fields do not share a shape")
    valid_prediction = np.isfinite(field["vx"]) & np.isfinite(field["vy"]) & (field["count"] > 0)
    available = reference_grid & valid_prediction
    error = np.hypot(field["vx"] - ref["vx"], field["vy"] - ref["vy"])
    limit = np.maximum(0.5, 0.2 * np.hypot(ref["vx"], ref["vy"]))
    total = int(reference_grid.sum())
    missing = int((reference_grid & ~valid_prediction).sum())
    bad = int((available & (error > limit)).sum())
    available_error = error[available]
    if total == 0:
        return {
            "reference_grid_size": 0,
            "prediction_coverage": float("nan"),
            "missing_vector_fraction": float("nan"),
            "failure_fraction": float("nan"),
            "epe_mean_available": float("nan"),
            "epe_median_available": float("nan"),
            "failure_missing": 0,
            "failure_error": 0,
        }
    return {
        "reference_grid_size": total,
        "prediction_coverage": float(available.sum()) / total,
        "missing_vector_fraction": float(missing) / total,
        "failure_fraction": float(missing + bad) / total,
        "epe_mean_available": float(available_error.mean()) if available_error.size else float("nan"),
        "epe_median_available": float(np.median(available_error)) if available_error.size else float("nan"),
        "failure_missing": missing,
        "failure_error": bad,
    }


def _cache_solver(
    original: Any,
    evt_path: Path,
    cfg_path: Path,
    output: Path,
    *,
    run_solver: bool,
) -> Tuple[Optional[np.ndarray], Optional[float], bool, Dict[str, Any]]:
    """Cache solver output only if its input and configuration hashes agree."""
    metadata_path = output.with_suffix(".solver.json")
    evt_hash, cfg_hash = sha256_file(evt_path), sha256_file(cfg_path)
    expected = {"evt_sha256": evt_hash, "cfg_sha256": cfg_hash, "solver": "original_corr_sum"}
    if output.is_file() and metadata_path.is_file():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(cached.get(key) == value for key, value in expected.items()):
                return np.load(output, allow_pickle=False)["arr"], 0.0, True, expected
        except (OSError, ValueError, KeyError):
            pass
    if not run_solver:
        return None, None, False, expected
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    array = original.run_ebiv(evt_path, output, cfg_path)
    elapsed = time.perf_counter() - started
    _dump_json(metadata_path, {**expected, "solver_wall_s": elapsed, "output": str(output)})
    return array, elapsed, False, expected


def _save_evt(original: Any, path: Path, event_dict: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    original.save_evt(path, dict(event_dict))
    return sha256_file(path)


def _checkpoint_tag(path: Path) -> str:
    return f"{path.stem}_{sha256_file(path)[:10]}"


def _unique_rates(values: Sequence[float]) -> List[float]:
    rates = [float(value) for value in values]
    if any(value < 0 for value in rates):
        raise ValueError("noise rates must be non-negative")
    if len(set(rates)) != len(rates):
        raise ValueError("noise rates must be unique to preserve RNG provenance")
    return rates


def _format_number(value: Any, digits: int = 5) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{float(value):.{digits}f}"


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run or prepare the reproducible EBIV gated benchmark requested by CLI."""
    original = _ebiv_module()
    evt_path, cfg_path = args.evt.expanduser().resolve(), args.cfg.expanduser().resolve()
    if not evt_path.is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"missing EBIV input: evt={evt_path}, cfg={cfg_path}")
    checkpoints = [Path(path).expanduser().resolve() for path in args.checkpoint]
    if not checkpoints:
        raise ValueError("at least one --checkpoint is required")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("each frozen checkpoint must appear at most once")
    if args.expect_training_seeds and len(checkpoints) != args.expect_training_seeds:
        raise ValueError(
            f"expected {args.expect_training_seeds} checkpoints, received {len(checkpoints)}; "
            "do not silently report a favorable subset"
        )
    rates = _unique_rates(args.noise_rate)
    out_dir = args.out_dir.expanduser().resolve()
    evt_dir, solver_dir = out_dir / "evt", out_dir / "solver"
    evt_dir.mkdir(parents=True, exist_ok=True)
    solver_dir.mkdir(parents=True, exist_ok=True)
    clean = original.load_evt(evt_path)
    height, width = [int(value) for value in clean["image_size"]]
    config = original.make_cfg(cfg_path)
    active_start, active_end = original.active_time_interval(config)
    source_hash, cfg_hash = sha256_file(evt_path), sha256_file(cfg_path)
    source_array = original.event_dict_to_array(clean)
    source_solver = solver_dir / f"source_{source_hash[:10]}_{cfg_hash[:10]}.npz"
    source_flow, source_s, source_reused, _ = _cache_solver(
        original, evt_path, cfg_path, source_solver, run_solver=args.run_solver
    )
    if source_flow is None and args.run_solver:
        raise RuntimeError("source EBIV solver did not produce an array")
    reference_grid = None
    if source_flow is not None:
        source_field = original.vector_fields(source_flow)
        reference_grid = (
            np.isfinite(source_field["vx"])
            & np.isfinite(source_field["vy"])
            & (source_field["count"] > 0)
        )

    rows: List[Dict[str, Any]] = []
    manifest_cases: List[Dict[str, Any]] = []
    source_active = (source_array[:, 0] >= active_start) & (source_array[:, 0] < active_end)
    source_row: Dict[str, Any] = {
        "rate": 0.0,
        "noise_seed": None,
        "noise_protocol": "none (recorded source stream)",
        "source_evt": str(evt_path),
        "source_evt_sha256": source_hash,
        "cfg": str(cfg_path),
        "cfg_sha256": cfg_hash,
        "method": "source_reference",
        "method_kind": "source",
        "event_path": str(evt_path),
        "event_sha256": source_hash,
        "solver_npz": str(source_solver),
        "solver_wall_s": source_s,
        "solver_reused": source_reused,
        "source_recall": 1.0,
        "noise_rejection": None,
        "retention": 1.0,
        "precision": 1.0,
        "n_source": int(source_active.sum()),
        "n_injected": 0,
        "active_interval_us": [int(active_start), int(active_end)],
        "reference_note": "source-event velocity map; source-derived reference, not absolute physical ground truth",
    }
    if source_flow is not None and reference_grid is not None:
        source_row.update(original.compute_metrics("source_reference", source_flow, source_flow))
        source_row.update(coverage_metrics(original, source_flow, source_flow, reference_grid))
    else:
        source_row.update(status="prepared_no_solver", epe_mean_px_per_ms=None, prediction_coverage=None, failure_fraction=None)
    rows.append(source_row)
    rng = np.random.default_rng(args.noise_seed)
    for rate in rates:
        noisy, labels = exact_labeled_noise(clean, rate, rng)
        noisy_array = original.event_dict_to_array(noisy)
        noisy_hash = sha256_array(noisy_array)
        rate_tag = f"r{int(round(rate * 100)):02d}"
        raw_evt = evt_dir / f"raw_{rate_tag}_{noisy_hash[:10]}.evt"
        raw_evt_hash = _save_evt(original, raw_evt, noisy)
        active_full = (noisy_array[:, 0] >= active_start) & (noisy_array[:, 0] < active_end)
        # Preserve the original prefix for raw and historical VGS.  A learned
        # context method can request a real future-event prefix so events near
        # active_end are not incorrectly treated as right-censored.
        prefix_full = noisy_array[:, 0] < active_end
        prefix_indices = np.flatnonzero(prefix_full)
        active_in_prefix = (noisy_array[prefix_indices, 0] >= active_start) & (
            noisy_array[prefix_indices, 0] < active_end
        )
        feature_lookahead_us = int(getattr(args, "feature_lookahead_us", 0) or 0)
        feature_prefix_full = noisy_array[:, 0] < active_end + feature_lookahead_us
        feature_prefix_indices = np.flatnonzero(feature_prefix_full)
        feature_active_in_prefix = (noisy_array[feature_prefix_indices, 0] >= active_start) & (
            noisy_array[feature_prefix_indices, 0] < active_end
        )
        if int(active_full.sum()) != int(active_in_prefix.sum()):
            raise RuntimeError("active-window index accounting drifted")
        if int(active_full.sum()) != int(feature_active_in_prefix.sum()):
            raise RuntimeError("feature active-window index accounting drifted")
        prefix = noisy_array[prefix_indices]
        feature_prefix = noisy_array[feature_prefix_indices]
        labels_active = labels[prefix_indices][active_in_prefix]
        case = {
            "rate": rate,
            "noise_seed": int(args.noise_seed),
            "noise_protocol": "legacy_rng_order_t_x_y_p_then_stable_time_sort",
            "source_evt": str(evt_path),
            "source_evt_sha256": source_hash,
            "cfg": str(cfg_path),
            "cfg_sha256": cfg_hash,
            "noisy_array_sha256": noisy_hash,
            "raw_evt": str(raw_evt),
            "raw_evt_sha256": raw_evt_hash,
            "active_interval_us": [int(active_start), int(active_end)],
            "n_source_total": int(len(clean["t"])),
            "n_injected_total": int(len(noisy["t"]) - len(clean["t"])),
            "n_active_total": int(active_in_prefix.sum()),
            "n_active_source": int(labels_active.sum()),
            "n_active_injected": int((~labels_active.astype(bool)).sum()),
            "feature_state_input": (
                "all raw chronological events with t < active_end + "
                f"{feature_lookahead_us} us" if feature_lookahead_us else
                "all raw chronological events with t < active_end"
            ),
            "selection_scope": "only events in [active_start, active_end); all others retained",
            "feature_lookahead_us": feature_lookahead_us,
            "feature_state_event_count": int(len(feature_prefix)),
        }
        manifest_cases.append(case)

        def append_solver_row(
            *,
            method: str,
            method_kind: str,
            event_path: Path,
            keep_active: np.ndarray,
            extra: Optional[Mapping[str, Any]] = None,
        ) -> None:
            event_hash = sha256_file(event_path)
            solver_path = solver_dir / f"{method}_{rate_tag}_{event_hash[:10]}_{cfg_hash[:10]}.npz"
            flow, solver_s, solver_reused, solver_provenance = _cache_solver(
                original, event_path, cfg_path, solver_path, run_solver=args.run_solver
            )
            row: Dict[str, Any] = {
                **case,
                "method": method,
                "method_kind": method_kind,
                "event_path": str(event_path),
                "event_sha256": event_hash,
                "solver_npz": str(solver_path),
                "solver_wall_s": solver_s,
                "solver_reused": solver_reused,
                "solver_provenance": solver_provenance,
                **_decision_metrics(labels_active, keep_active),
            }
            if flow is not None and source_flow is not None and reference_grid is not None:
                row.update(original.compute_metrics(method, flow, source_flow))
                row.update(coverage_metrics(original, flow, source_flow, reference_grid))
            else:
                row.update(
                    status="prepared_no_solver",
                    epe_mean_px_per_ms=None,
                    epe_median_px_per_ms=None,
                    prediction_coverage=None,
                    failure_fraction=None,
                )
            if extra:
                row.update(dict(extra))
            rows.append(row)

        # Raw baseline: same noisy stream, no selection.
        append_solver_row(
            method="raw_noisy",
            method_kind="raw",
            event_path=raw_evt,
            keep_active=np.ones(active_in_prefix.sum(), dtype=bool),
        )

        # The untouched original support filter is recreated with its original
        # active-only policy and defaults.  This is the historical non-network
        # baseline; its result is written in the new run directory.
        support_event, support_stats = original.denoise_support_active(
            noisy,
            active_start,
            active_end,
            radius=args.support_radius,
            tau_us=args.support_tau_us,
            min_neighbors=args.support_min_neighbors,
        )
        support_prefix_mask = original.temporal_support_mask(
            prefix,
            [height, width],
            radius=args.support_radius,
            tau_us=args.support_tau_us,
            min_neighbors=args.support_min_neighbors,
        )
        support_keep_active = support_prefix_mask[active_in_prefix]
        if int(support_keep_active.sum()) != int(support_stats["active_retained_events"]):
            raise RuntimeError("historical support baseline selection no longer matches original function")
        support_evt = evt_dir / f"historical_support_{rate_tag}_{noisy_hash[:10]}.evt"
        _save_evt(original, support_evt, support_event)
        append_solver_row(
            method="historical_support_vgs",
            method_kind="historical_support",
            event_path=support_evt,
            keep_active=support_keep_active,
            extra={"historical_support_stats": support_stats},
        )

        # New frozen gated methods.  Features are causal over exactly the raw
        # prefix t < active_end, and only the active interval is selected.
        for checkpoint_path in checkpoints:
            started = time.perf_counter()
            inference = infer_gated_from_checkpoint(
                feature_prefix,
                height,
                width,
                checkpoint_path,
                batch_size=args.batch_size,
                return_features=False,
            )
            inference_s = time.perf_counter() - started
            keep_active = np.asarray(inference["keep"], dtype=bool)[feature_active_in_prefix]
            keep_all = np.ones(len(noisy_array), dtype=bool)
            keep_all[prefix_indices[active_in_prefix]] = keep_active
            gated_event = original.subset_event_dict(noisy, keep_all)
            tag = _checkpoint_tag(checkpoint_path)
            learned_prefix = str(args.learned_method_prefix)
            learned_kind = str(args.learned_method_kind)
            gated_evt = evt_dir / f"{learned_prefix}_{tag}_{rate_tag}_{noisy_hash[:10]}.evt"
            gated_evt_hash = _save_evt(original, gated_evt, gated_event)
            alpha_value = inference.get("alpha")
            active_alpha = (np.asarray(alpha_value)[feature_active_in_prefix]
                            if alpha_value is not None else np.empty(0, dtype=np.float32))
            physics_value = inference.get("p_physics")
            ai_value = inference.get("p_ai")
            active_p_physics = (np.asarray(physics_value)[feature_active_in_prefix]
                                if physics_value is not None else np.empty(0, dtype=np.float32))
            active_p_ai = (np.asarray(ai_value)[feature_active_in_prefix]
                           if ai_value is not None else np.empty(0, dtype=np.float32))
            append_solver_row(
                method=f"{learned_prefix}_{tag}",
                method_kind=learned_kind,
                event_path=gated_evt,
                keep_active=keep_active,
                extra={
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": inference["checkpoint_sha256"],
                    "checkpoint_schema": inference["checkpoint_schema"],
                    "training_seed": inference.get("training_seed"),
                    "threshold": inference["threshold"],
                    "inference_wall_s": inference_s,
                    "gated_evt_sha256": gated_evt_hash,
                    "mean_alpha_active": float(active_alpha.mean()) if active_alpha.size else None,
                    "std_alpha_active": float(active_alpha.std()) if active_alpha.size else None,
                    "mean_expert_disagreement_active": float(
                        np.abs(active_p_physics - active_p_ai).mean()
                    ) if active_p_physics.size and active_p_ai.size else None,
                    "learned_inference_metadata": inference.get("metadata"),
                },
            )

    fields = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = out_dir / f"{args.result_stem}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value, ensure_ascii=False, default=_json_default) if isinstance(value, (dict, list)) else value
                 for key, value in row.items()}
            )
    summary: Dict[str, Any] = {}
    for rate in rates:
        rate_rows = [row for row in rows if row["rate"] == rate and row["method_kind"] == str(args.learned_method_kind)]
        for metric in ("epe_mean_px_per_ms", "prediction_coverage", "failure_fraction", "source_recall", "noise_rejection"):
            values = [row.get(metric) for row in rate_rows if row.get(metric) is not None and np.isfinite(row[metric])]
            if values:
                summary[f"rate_{rate:g}_{metric}"] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n_checkpoints": len(values),
                }
    result = {
        "protocol": str(args.protocol_name),
        "status": "completed" if args.run_solver else "prepared_no_solver",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
        "source": {"evt": str(evt_path), "evt_sha256": source_hash, "cfg": str(cfg_path), "cfg_sha256": cfg_hash},
        "active_interval_us": [int(active_start), int(active_end)],
        "noise": {"rates": rates, "seed": int(args.noise_seed), "exact_rng_order": "t,x,y,p; stable timestamp sort"},
        "checkpoints": [
            {"path": str(path), "sha256": sha256_file(path)} for path in checkpoints
        ],
        "reference": "source-event EBIV velocity map; source-derived reference, not absolute physical ground truth",
        "selection": str(args.selection_note),
        "historical_baseline": {
            "method": "denoise_support_active from untouched ebiv_noise_vector_experiment.py",
            "radius": int(args.support_radius),
            "tau_us": int(args.support_tau_us),
            "min_neighbors": int(args.support_min_neighbors),
        },
        "cases": manifest_cases,
        "rows": rows,
        "summary": summary,
    }
    _dump_json(out_dir / f"{args.result_stem}_metadata.json", result)
    report = [
        f"# {args.report_title}",
        "",
        "The source-event EBIV map is a source-derived reference, not an independent physical ground truth. "
        "All source-reference grid positions count in coverage and failure; missing vectors never lower EPE by omission.",
        "",
        "| Noise | Method | EPE (px/ms) | Coverage | Failure | Source recall | Noise rejection |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {100 * row['rate']:.0f}% | {row['method']} | "
            f"{_format_number(row.get('epe_mean_px_per_ms'), 4)} | "
            f"{_format_number(row.get('prediction_coverage'), 4)} | "
            f"{_format_number(row.get('failure_fraction'), 4)} | "
            f"{_format_number(row.get('source_recall'), 4)} | "
            f"{_format_number(row.get('noise_rejection'), 4)} |"
        )
    report.extend(
        [
            "",
            "`historical_support_vgs` is the original non-network temporal-support baseline. "
            f"Each `{args.learned_method_prefix}_*` row uses a separately frozen checkpoint and its validation-selected threshold.",
        ]
    )
    (out_dir / f"{args.result_stem}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen gated dual-expert event selection on the original EBIV noise benchmark."
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="one or three frozen gated checkpoint files")
    parser.add_argument("--evt", type=Path, default=DEFAULT_EVT, help="original EVT source recording")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="original EBIV parameter file")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="new output directory; historical files are never modified")
    parser.add_argument("--noise-rate", type=float, nargs="+", default=[0.10, 0.20, 0.40], help="rates in original execution order")
    parser.add_argument("--noise-seed", type=int, default=20260605, help="original benchmark RNG seed")
    parser.add_argument("--support-radius", type=int, default=2, help="original support-VGS radius")
    parser.add_argument("--support-tau-us", type=int, default=5000, help="original support-VGS temporal window")
    parser.add_argument("--support-min-neighbors", type=int, default=1, help="original support-VGS support count")
    parser.add_argument("--batch-size", type=int, default=65536, help="CPU inference batch size")
    parser.add_argument("--feature-lookahead-us", type=int, default=0,
                        help="real future-event prefix for learned context features; default 0 preserves legacy behavior")
    parser.add_argument("--expect-training-seeds", type=int, default=3, help="required checkpoint count; set 0 only for a smoke test")
    parser.add_argument("--run-solver", action="store_true", help="run original corr_sum EBIV solver; without it, write streams and provenance only")
    parser.add_argument("--learned-method-prefix", default="gated", help="filename/method prefix for the learned checkpoint rows")
    parser.add_argument("--learned-method-kind", default="gated_dual_expert", help="semantic method kind used in rows and summaries")
    parser.add_argument("--result-stem", default="ebiv_gated", help="basename for CSV, metadata, and Markdown outputs")
    parser.add_argument("--protocol-name", default="original_ebiv_uniform_noise_with_frozen_gated_dual_expert")
    parser.add_argument("--selection-note", default="features/state use raw chronological events with t < active_end; masks affect active interval only")
    parser.add_argument("--report-title", default="Frozen gated dual-expert EBIV benchmark")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "out_dir": str(args.out_dir.expanduser().resolve()),
                "n_rows": len(result["rows"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
