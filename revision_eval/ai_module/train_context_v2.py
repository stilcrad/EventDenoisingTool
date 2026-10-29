"""Train the fixed-latency contextual AI extension of the VGS front end.

This is a separate follow-up to ``gated_v1``.  It never overwrites historical
VGS data and never treats the VGS-derived calibrated score as the historical
full VGS implementation.  Training and model/threshold selection use only the
train and validation scenes in ``context_v2_protocol.json``.  Held-out hot-
plate windows are materialised only by the explicit ``evaluate`` action.

The context expert operates on original events, not accumulated frames.  Its
100-D time-surface descriptor uses exactly 5 ms of future context, so the
method has a declared fixed 5 ms latency rather than zero-delay causality.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

import experiment as base
from features import extract as extract_physics
from features_context import LOOKAHEAD_US, complete_context_mask, extract_context
from gated_model import PhysicsCalibrator, fit_platt, set_cpu_seed


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "context_v2_protocol.json"
RUN = HERE / "runs" / "context_v2"
EPS = 1.0e-7


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cfg():
    value = read_json(PROTOCOL_PATH)
    expected = "pre-registered before context-v2 held-out event and mechanics evaluation"
    if value.get("status") != expected:
        raise RuntimeError("context-v2 protocol is not in its frozen pre-evaluation state")
    if int(value["lookahead_us"]) != int(LOOKAHEAD_US):
        raise RuntimeError("protocol lookahead and feature implementation differ")
    groups = [set(value["splits"][name]) for name in ("train", "validation", "test")]
    if any(left.intersection(right) for i, left in enumerate(groups) for right in groups[i + 1:]):
        raise RuntimeError("train/validation/test scenes overlap")
    return value


def freeze_protocol(c) -> None:
    target = RUN / "frozen_protocol.json"
    if target.exists() and read_json(target) != c:
        raise RuntimeError("existing frozen context-v2 protocol differs")
    if not target.exists():
        write_json(target, c)


def effective_seed(split: str, base_seed: int, scene_i: int, fraction_i: int, corruption_i: int) -> int:
    offsets = {"train": 0, "validation": 10_000_000, "test": 20_000_000}
    return int(base_seed + offsets[split] + 100_000 * scene_i + 1_000 * fraction_i + 10 * corruption_i)


def load_source(scene: str, fraction: float, c):
    """Load warmup, active interval, and the declared post-window lookahead."""
    path = base.DATA / scene / "prophesee_0" / "events.hdf5"
    warmup = int(c["warmup_us"])
    lookahead = int(c["lookahead_us"])
    with h5py.File(path, "r") as handle:
        group = handle["raw_events"]
        first, last = int(group["t"][0]), int(group["t"][-1])
        start = int(first + float(fraction) * (last - first))
        stop = start + int(c["window_us"])
        if start - warmup < first or stop + lookahead > last:
            raise ValueError(f"context window outside recording for {scene} at {fraction:g}")
        i0 = base.search(group["t"], start - warmup)
        i1 = base.search(group["t"], stop + lookahead)
        events = np.column_stack([group[name][i0:i1] for name in ("t", "x", "y", "p")]).astype(np.int64)
    events[:, 3] = events[:, 3] > 0
    info = {
        "source": str(path),
        "source_bytes": int(path.stat().st_size),
        "source_mtime_ns": int(path.stat().st_mtime_ns),
        "source_indices": [int(i0), int(i1)],
        "source_slice_sha256": base.hash_array(events),
        "active_window_us": [int(start), int(stop)],
        "loaded_window_us": [int(start - warmup), int(stop + lookahead)],
    }
    return events, start, stop, info


def balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    pieces = []
    for cls in (0, 1):
        available = np.flatnonzero(labels == cls)
        if not len(available):
            raise RuntimeError(f"case contains no proxy-label class {cls}")
        take = min(int(per_class), int(len(available)))
        pieces.append(rng.choice(available, take, replace=False))
    return np.sort(np.concatenate(pieces)).astype(np.int64)


def materialize_case(split: str, scene: str, fraction: float, kind: str, rate: float,
                     base_seed: int, scene_i: int, fraction_i: int, corruption_i: int, c):
    source, start, stop, info = load_source(scene, fraction, c)
    seed = effective_seed(split, base_seed, scene_i, fraction_i, corruption_i)
    noisy, labels = base.inject(source, rate, kind, seed, int(c["sensor"]["height"]),
                                int(c["sensor"]["width"]), 80)
    t0 = time.perf_counter()
    physics, scores = extract_physics(noisy, int(c["sensor"]["height"]), int(c["sensor"]["width"]))
    physics_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    context = extract_context(noisy, int(c["sensor"]["height"]), int(c["sensor"]["width"]))
    context_seconds = time.perf_counter() - t1
    complete = complete_context_mask(noisy[:, 0])
    active = (noisy[:, 0] >= start) & (noisy[:, 0] < stop) & complete
    if not active.any():
        raise RuntimeError(f"no complete active events for {scene}/{fraction:g}/{kind}/{rate:g}")
    active_labels = labels[active]
    meta = {
        "split": split,
        "scene": scene,
        "fraction": float(fraction),
        "noise_type": kind,
        "rate": float(rate),
        "base_noise_seed": int(base_seed),
        "effective_noise_seed": int(seed),
        "n_source_loaded": int(len(source)),
        "n_noisy_loaded": int(len(noisy)),
        "n_active": int(active.sum()),
        "n_active_source": int((active_labels == 1).sum()),
        "n_active_injected": int((active_labels == 0).sum()),
        "physics_feature_seconds": float(physics_seconds),
        "context_feature_seconds": float(context_seconds),
        "fixed_latency_us": int(c["lookahead_us"]),
        "noisy_sha256": base.hash_array(noisy),
        **info,
    }
    return physics[active], context[active], scores[active], active_labels, meta


def case_key(scene, fraction, kind, rate, seed):
    return f"{scene}_f{fraction:g}_{kind}_r{rate:g}_seed{seed}"


def prepare_split(split: str, c):
    if split not in {"train", "validation"}:
        raise ValueError("prepare may only materialise train/validation data")
    records = []
    per_class = int(c["sampling"][f"{split}_per_class_per_case"])
    for scene_i, scene in enumerate(c["splits"][split]):
        for fraction_i, fraction in enumerate(c["window_fractions"][split]):
            for corruption_i, (kind, rate) in enumerate(c["corruptions"][split]):
                for base_seed in c["noise_seeds"][split]:
                    key = case_key(scene, fraction, kind, rate, base_seed)
                    cache = RUN / "cache" / split / f"{key}.npz"
                    meta_path = cache.with_suffix(".json")
                    if cache.exists() and meta_path.exists():
                        record = read_json(meta_path)
                        if record.get("protocol_sha256") != file_hash(PROTOCOL_PATH):
                            raise RuntimeError(f"stale cache protocol for {cache}")
                        records.append(record)
                        continue
                    physics, context, scores, labels, meta = materialize_case(
                        split, scene, fraction, kind, rate, base_seed,
                        scene_i, fraction_i, corruption_i, c,
                    )
                    chosen = balanced_indices(labels, per_class, int(meta["effective_noise_seed"]) + 900_000)
                    sampled_labels = labels[chosen]
                    if split == "validation":
                        selected = balanced_indices(
                            sampled_labels,
                            int(c["sampling"]["validation_selection_per_class"]),
                            int(meta["effective_noise_seed"]) + 1_900_000,
                        )
                    else:
                        selected = np.arange(len(chosen), dtype=np.int64)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache,
                        physics=physics[chosen].astype(np.float32),
                        context=context[chosen].astype(np.float32),
                        scores=scores[chosen].astype(np.float32),
                        labels=sampled_labels.astype(np.uint8),
                        selection_indices=selected,
                    )
                    meta.update({
                        "cache": str(cache),
                        "cache_sha256": file_hash(cache),
                        "protocol_sha256": file_hash(PROTOCOL_PATH),
                        "cached_events": int(len(chosen)),
                        "cached_source": int((sampled_labels == 1).sum()),
                        "cached_injected": int((sampled_labels == 0).sum()),
                        "selection_events": int(len(selected)),
                    })
                    write_json(meta_path, meta)
                    records.append(meta)
                    print(f"prepared {split} {key}: active={len(labels):,}, cached={len(chosen):,}", flush=True)
    records.sort(key=lambda row: (row["scene"], row["fraction"], row["noise_type"], row["rate"], row["base_noise_seed"]))
    write_json(RUN / f"{split}_manifest.json", records)
    return records


def cache_records(split: str):
    path = RUN / f"{split}_manifest.json"
    if not path.exists():
        raise RuntimeError(f"missing {path}; run prepare first")
    return read_json(path)


def load_cache(record):
    with np.load(record["cache"]) as data:
        return tuple(np.asarray(data[name]) for name in ("physics", "context", "scores", "labels", "selection_indices"))


def concatenate(records):
    loaded = [load_cache(record) for record in records]
    return tuple(np.concatenate([part[index] for part in loaded]) for index in range(4))


def norm_fit(array: np.ndarray):
    mean = array.mean(axis=0).astype(np.float32)
    scale = np.maximum(array.std(axis=0), 1.0e-6).astype(np.float32)
    return mean, scale


def standardize(array: np.ndarray, norm):
    mean, scale = norm
    return np.clip((array - mean) / scale, -20.0, 20.0).astype(np.float32)


class TinyMLP(nn.Module):
    def __init__(self, width: int, hidden=(64, 32)):
        super().__init__()
        layers = []
        previous = int(width)
        for current in hidden:
            layers.extend((nn.Linear(previous, int(current)), nn.ReLU()))
            previous = int(current)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def probabilities(model: nn.Module, array: np.ndarray, batch_size=65_536):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            logits = model(torch.from_numpy(array[start:start + batch_size]))
            output.append(torch.sigmoid(logits).reshape(-1).numpy())
    return np.concatenate(output) if output else np.empty(0, np.float32)


def calibrated_probabilities(calibrator: PhysicsCalibrator, scores: np.ndarray, batch_size=65_536):
    output = []
    calibrator.eval()
    with torch.no_grad():
        for start in range(0, len(scores), batch_size):
            output.append(calibrator(scores[start:start + batch_size]).reshape(-1).numpy())
    return np.concatenate(output) if output else np.empty(0, np.float32)


def selection_cases(records, context_norm, joint_norm):
    context_cases, joint_cases, raw_cases = [], [], []
    for record in records:
        physics, context, scores, labels, selected = load_cache(record)
        joint = np.concatenate((physics, context), axis=1)
        context_cases.append((standardize(context[selected], context_norm), labels[selected]))
        joint_cases.append((standardize(joint[selected], joint_norm), labels[selected]))
        raw_cases.append((physics[selected], context[selected], scores[selected], labels[selected]))
    return context_cases, joint_cases, raw_cases


def macro_auc(model, cases):
    return float(np.mean([roc_auc_score(labels, probabilities(model, features)) for features, labels in cases]))


def train_binary(model, x_train, labels, validation_cases, c, seed, label, epochs=None):
    set_cpu_seed(int(seed))
    torch.set_num_threads(4)
    model.train()
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(labels.astype(np.float32))[:, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(c["optimization"]["learning_rate"]),
                                 weight_decay=float(c["optimization"]["weight_decay"]))
    epochs = int(epochs or c["optimization"]["epochs"])
    best_auc, best_state, best_epoch, history = -math.inf, None, 0, []
    start_time = time.perf_counter()
    generator = torch.Generator().manual_seed(int(seed) + 17)
    for epoch in range(epochs):
        order = torch.randperm(len(x_tensor), generator=generator)
        total = 0.0
        model.train()
        for start in range(0, len(order), int(c["optimization"]["batch_size"])):
            index = order[start:start + int(c["optimization"]["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x_tensor[index]), y_tensor[index])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        score = macro_auc(model, validation_cases)
        history.append({"epoch": epoch + 1, "train_bce": total / len(order), "macro_validation_auc": score})
        if score > best_auc:
            best_auc, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    print(f"trained {label}_{seed}: validation macro AUC={best_auc:.5f}, epoch={best_epoch}", flush=True)
    return model, {"best_epoch": best_epoch, "macro_validation_auc": best_auc,
                   "training_seconds": time.perf_counter() - start_time, "history": history}


def gate_input(physics_std, context_std, p_physics, p_ai):
    return np.concatenate((physics_std, context_std, p_physics[:, None], p_ai[:, None],
                           np.abs(p_physics - p_ai)[:, None]), axis=1).astype(np.float32)


def gated_probabilities(gate, features, p_physics, p_ai, batch_size=65_536):
    gate.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            alpha = torch.sigmoid(gate(torch.from_numpy(features[start:start + batch_size]))).reshape(-1).numpy()
            output.append((1.0 - alpha) * p_physics[start:start + batch_size] + alpha * p_ai[start:start + batch_size])
    return np.concatenate(output) if output else np.empty(0, np.float32)


def train_gate(gate, features, labels, p_physics, p_ai, validation_cases, c, seed, epochs=None):
    set_cpu_seed(int(seed) + 101)
    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(labels.astype(np.float32))
    pp_tensor = torch.from_numpy(p_physics.astype(np.float32))
    pa_tensor = torch.from_numpy(p_ai.astype(np.float32))
    optimizer = torch.optim.Adam(gate.parameters(), lr=float(c["optimization"]["learning_rate"]),
                                 weight_decay=float(c["optimization"]["weight_decay"]))
    epochs = int(epochs or c["optimization"]["epochs"])
    best_auc, best_state, best_epoch, history = -math.inf, None, 0, []
    generator = torch.Generator().manual_seed(int(seed) + 119)
    started = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(len(x_tensor), generator=generator)
        total = 0.0
        gate.train()
        for start in range(0, len(order), int(c["optimization"]["batch_size"])):
            index = order[start:start + int(c["optimization"]["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            alpha = torch.sigmoid(gate(x_tensor[index])).reshape(-1)
            fused = ((1.0 - alpha) * pp_tensor[index] + alpha * pa_tensor[index]).clamp(EPS, 1.0 - EPS)
            loss = nn.functional.binary_cross_entropy(fused, y_tensor[index])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        values = [roc_auc_score(y, gated_probabilities(gate, x, pp, pa)) for x, pp, pa, y in validation_cases]
        score = float(np.mean(values))
        history.append({"epoch": epoch + 1, "train_bce": total / len(order), "macro_validation_auc": score})
        if score > best_auc:
            best_auc, best_state, best_epoch = score, copy.deepcopy(gate.state_dict()), epoch + 1
    gate.load_state_dict(best_state)
    print(f"trained gated_context_dual_expert_{seed}: validation macro AUC={best_auc:.5f}, epoch={best_epoch}", flush=True)
    return gate, {"best_epoch": best_epoch, "macro_validation_auc": best_auc,
                  "training_seconds": time.perf_counter() - started, "history": history}


def operating_threshold(probability_cases, label_cases, mean_recall=0.98, minimum_recall=0.95):
    positives = [probabilities_[labels == 1] for probabilities_, labels in zip(probability_cases, label_cases)]
    low = min(float(values.min()) for values in positives) - EPS
    high = max(float(values.max()) for values in positives) + EPS
    def acceptable(value):
        recalls = np.asarray([(scores >= value).mean() for scores in positives], np.float64)
        return float(recalls.mean()) >= mean_recall and float(recalls.min()) >= minimum_recall
    for _ in range(50):
        middle = (low + high) / 2.0
        if acceptable(middle):
            low = middle
        else:
            high = middle
    recalls = np.asarray([(scores >= low).mean() for scores in positives], np.float64)
    return float(low), {"mean_validation_source_recall": float(recalls.mean()),
                        "minimum_validation_source_recall": float(recalls.min())}


def metrics(labels, score, threshold):
    labels = np.asarray(labels, np.uint8)
    score = np.asarray(score, np.float32)
    keep = score >= float(threshold)
    source, injected = labels == 1, labels == 0
    tp, fp = int((source & keep).sum()), int((injected & keep).sum())
    return {
        "source_recall": tp / max(int(source.sum()), 1),
        "injected_noise_rejection": 1.0 - fp / max(int(injected.sum()), 1),
        "retention": float(keep.mean()),
        "precision": tp / max(tp + fp, 1),
        "average_precision": float(average_precision_score(labels, score)),
        "roc_auc": float(roc_auc_score(labels, score)),
    }


def build_training(seed, c, train_records, validation_records, epochs=None):
    physics, context, scores, labels = concatenate(train_records)
    context_norm = norm_fit(context)
    physics_norm = norm_fit(physics)
    joint = np.concatenate((physics, context), axis=1)
    joint_norm = norm_fit(joint)
    x_context = standardize(context, context_norm)
    x_physics = standardize(physics, physics_norm)
    x_joint = standardize(joint, joint_norm)
    context_cases, joint_cases, raw_validation = selection_cases(validation_records, context_norm, joint_norm)
    calibrator = fit_platt(scores, labels, class_balanced=True, max_iter=200, seed=7002).freeze()
    set_cpu_seed(int(seed))
    context_model, context_info = train_binary(TinyMLP(100), x_context, labels, context_cases, c, seed,
                                                "context_ai_expert", epochs)
    set_cpu_seed(int(seed) + 1)
    joint_model, joint_info = train_binary(TinyMLP(114), x_joint, labels, joint_cases, c, seed,
                                            "physics_context_mlp", epochs)
    p_physics = calibrated_probabilities(calibrator, scores)
    p_ai = probabilities(context_model, x_context)
    x_gate = gate_input(x_physics, x_context, p_physics, p_ai)
    gate_validation = []
    for raw in raw_validation:
        vp, vc, vs, vy = raw
        vps, vcs = standardize(vp, physics_norm), standardize(vc, context_norm)
        vpp = calibrated_probabilities(calibrator, vs)
        vpa = probabilities(context_model, vcs)
        gate_validation.append((gate_input(vps, vcs, vpp, vpa), vpp, vpa, vy))
    set_cpu_seed(int(seed) + 2)
    gate, gate_info = train_gate(TinyMLP(117, hidden=(64, 16)), x_gate, labels, p_physics, p_ai,
                                 gate_validation, c, seed, epochs)
    validation_predictions = {"context_ai_expert": [], "physics_context_mlp": [],
                              "gated_context_dual_expert": [], "physics_calibrated": []}
    validation_labels = []
    validation_rows = []
    for record in validation_records:
        vp, vc, vs, vy, _ = load_cache(record)
        vcs = standardize(vc, context_norm)
        vps = standardize(vp, physics_norm)
        vjoint = standardize(np.concatenate((vp, vc), axis=1), joint_norm)
        vpp = calibrated_probabilities(calibrator, vs)
        vpa = probabilities(context_model, vcs)
        predictions = {
            "context_ai_expert": vpa,
            "physics_context_mlp": probabilities(joint_model, vjoint),
            "gated_context_dual_expert": gated_probabilities(gate, gate_input(vps, vcs, vpp, vpa), vpp, vpa),
            "physics_calibrated": vpp,
        }
        validation_labels.append(vy)
        for name, value in predictions.items():
            validation_predictions[name].append(value)
    thresholds, threshold_info = {}, {}
    for name, cases in validation_predictions.items():
        threshold, info = operating_threshold(cases, validation_labels)
        thresholds[name], threshold_info[name] = threshold, info
    selection_scores = {
        "context_ai_expert": float(context_info["macro_validation_auc"]),
        "physics_context_mlp": float(joint_info["macro_validation_auc"]),
        "gated_context_dual_expert": float(gate_info["macro_validation_auc"]),
    }
    tie_order = {"gated_context_dual_expert": 2, "physics_context_mlp": 1, "context_ai_expert": 0}
    primary = max(selection_scores, key=lambda name: (selection_scores[name], tie_order[name]))
    for record, vy, predictions in zip(validation_records, validation_labels,
                                        zip(*[validation_predictions[name] for name in validation_predictions])):
        identity = {key: record[key] for key in ("scene", "fraction", "noise_type", "rate", "base_noise_seed", "effective_noise_seed")}
        for name, value in zip(validation_predictions, predictions):
            validation_rows.append({"split": "validation", "training_seed": int(seed), "method": name,
                                    "threshold": thresholds[name], **identity, **metrics(vy, value, thresholds[name])})
    result = {
        "seed": int(seed), "primary_method": primary, "selection_scores": selection_scores,
        "thresholds": thresholds, "threshold_info": threshold_info,
        "training": {"context_ai_expert": context_info, "physics_context_mlp": joint_info,
                     "gated_context_dual_expert": gate_info},
        "calibrator": calibrator, "context_model": context_model, "joint_model": joint_model, "gate": gate,
        "context_norm": context_norm, "physics_norm": physics_norm, "joint_norm": joint_norm,
        "validation_rows": validation_rows,
    }
    return result


def serializable_checkpoint(result, c):
    def norm_value(value):
        return {"mean": value[0], "scale": value[1]}
    return {
        "schema": "vgs_physics_ai_contextual_dual_expert_v2",
        "seed": result["seed"],
        "primary_method": result["primary_method"],
        "selection_scores": result["selection_scores"],
        "thresholds": result["thresholds"],
        "threshold_info": result["threshold_info"],
        "context_ai_state": result["context_model"].state_dict(),
        "physics_context_state": result["joint_model"].state_dict(),
        "gate_state": result["gate"].state_dict(),
        "physics_calibration_state": result["calibrator"].state_dict(),
        "physics_calibration": result["calibrator"].calibration_dict(),
        "context_normalization": norm_value(result["context_norm"]),
        "physics_normalization": norm_value(result["physics_norm"]),
        "joint_normalization": norm_value(result["joint_norm"]),
        "fixed_latency_us": int(c["lookahead_us"]),
        "label_boundary": c["label_boundary"],
        "reference_boundary": c["reference_boundary"],
        "negative_result_boundary": c["negative_result_boundary"],
    }


def run_probe(c):
    result = build_training(int(c["training_seeds"][0]), c, cache_records("train"), cache_records("validation"), epochs=8)
    report = {key: result[key] for key in ("seed", "primary_method", "selection_scores", "thresholds", "threshold_info")}
    operating_point = {}
    for method in ("context_ai_expert", "physics_context_mlp", "gated_context_dual_expert", "physics_calibrated"):
        method_rows = [row for row in result["validation_rows"] if row["method"] == method]
        grouped = {"all": method_rows}
        for noise_type in ("uniform", "hot_pixel"):
            grouped[noise_type] = [row for row in method_rows if row["noise_type"] == noise_type]
        operating_point[method] = {
            group: {
                "n_cases": len(rows),
                "mean_source_recall": float(np.mean([row["source_recall"] for row in rows])),
                "mean_injected_noise_rejection": float(np.mean([row["injected_noise_rejection"] for row in rows])),
                "mean_retention": float(np.mean([row["retention"] for row in rows])),
            }
            for group, rows in grouped.items() if rows
        }
    report["validation_operating_point"] = operating_point
    report["status"] = "train-validation probe only; no test or mechanics data read"
    write_json(RUN / "probe_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def run_train(c):
    train_records, validation_records = cache_records("train"), cache_records("validation")
    manifest, rows = [], []
    model_dir = RUN / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed in c["training_seeds"]:
        result = build_training(int(seed), c, train_records, validation_records)
        checkpoint = model_dir / f"context_v2_{seed}.pt"
        torch.save(serializable_checkpoint(result, c), checkpoint)
        rows.extend(result["validation_rows"])
        manifest.append({"seed": int(seed), "checkpoint": str(checkpoint),
                         "checkpoint_sha256": file_hash(checkpoint),
                         "primary_method": result["primary_method"],
                         "selection_scores": result["selection_scores"],
                         "thresholds": result["thresholds"]})
        print(f"frozen context-v2 checkpoint {seed}; primary={result['primary_method']}", flush=True)
    write_csv(RUN / "validation_metrics.csv", rows)
    write_json(RUN / "frozen_manifest.json", {
        "protocol_sha256": file_hash(PROTOCOL_PATH),
        "models": manifest,
        "train_manifest_sha256": file_hash(RUN / "train_manifest.json"),
        "validation_manifest_sha256": file_hash(RUN / "validation_manifest.json"),
        "selection_data": "train and validation only",
        "test_evaluation_started": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": "cpu"},
    })


def restore(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Keep the v2 inference path backward compatible with the independently
    # frozen v3 reliability-network checkpoints.  The tensor layout is
    # intentionally unchanged, while the checkpoint schema records the
    # different training objective and protocol.
    accepted_schemas = {
        "vgs_physics_ai_contextual_dual_expert_v2",
        "vgs_physics_ai_contextual_reliability_v3",
        "vgs_physics_ai_contextual_reliability_v3_measurement_corrected",
        "vgs_physics_ai_contextual_reliability_v4_augmented",
    }
    if payload.get("schema") not in accepted_schemas:
        raise ValueError(f"unexpected checkpoint schema in {path}")
    context_model, joint_model, gate = TinyMLP(100), TinyMLP(114), TinyMLP(117, hidden=(64, 16))
    context_model.load_state_dict(payload["context_ai_state"])
    joint_model.load_state_dict(payload["physics_context_state"])
    gate.load_state_dict(payload["gate_state"])
    calibration = payload["physics_calibration"]
    calibrator = PhysicsCalibrator(eps=float(calibration["eps"]))
    calibrator.load_state_dict(payload["physics_calibration_state"])
    calibrator.freeze()
    def norm(name):
        return (np.asarray(payload[name]["mean"], np.float32), np.asarray(payload[name]["scale"], np.float32))
    return payload, calibrator, context_model.eval(), joint_model.eval(), gate.eval(), \
        norm("context_normalization"), norm("physics_normalization"), norm("joint_normalization")


def predict_all(restored, physics, context, scores):
    payload, calibrator, context_model, joint_model, gate, context_norm, physics_norm, joint_norm = restored
    context_std = standardize(context, context_norm)
    physics_std = standardize(physics, physics_norm)
    joint_std = standardize(np.concatenate((physics, context), axis=1), joint_norm)
    p_physics = calibrated_probabilities(calibrator, scores)
    p_ai = probabilities(context_model, context_std)
    return {
        "context_ai_expert": p_ai,
        "physics_context_mlp": probabilities(joint_model, joint_std),
        "gated_context_dual_expert": gated_probabilities(
            gate, gate_input(physics_std, context_std, p_physics, p_ai), p_physics, p_ai),
        "physics_calibrated": p_physics,
    }


def run_evaluate(c):
    manifest_path = RUN / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("missing frozen model manifest; run train first")
    manifest = read_json(manifest_path)
    if manifest["protocol_sha256"] != file_hash(PROTOCOL_PATH):
        raise RuntimeError("protocol changed after model freeze")
    paths = [Path(row["checkpoint"]) for row in manifest["models"]]
    rows, cases, runtimes = [], [], []
    for scene_i, scene in enumerate(c["splits"]["test"]):
        for fraction_i, fraction in enumerate(c["window_fractions"]["test"]):
            for corruption_i, (kind, rate) in enumerate(c["corruptions"]["test"]):
                for base_seed in c["noise_seeds"]["test"]:
                    physics, context, scores, labels, meta = materialize_case(
                        "test", scene, fraction, kind, rate, base_seed,
                        scene_i, fraction_i, corruption_i, c,
                    )
                    identity = {key: meta[key] for key in ("scene", "fraction", "noise_type", "rate", "base_noise_seed", "effective_noise_seed", "n_active", "fixed_latency_us", "noisy_sha256")}
                    rows.append({"split": "test", "training_seed": "", "method": "raw",
                                 "threshold": 0.0, **identity, **metrics(labels, np.ones(len(labels), np.float32), 0.0)})
                    cases.append(meta)
                    for path in paths:
                        restored = restore(path)
                        payload = restored[0]
                        started = time.perf_counter()
                        predictions = predict_all(restored, physics, context, scores)
                        elapsed = time.perf_counter() - started
                        for name, values in predictions.items():
                            threshold = float(payload["thresholds"][name])
                            rows.append({"split": "test", "training_seed": int(payload["seed"]),
                                         "primary_method": payload["primary_method"], "method": name,
                                         "threshold": threshold, **identity, **metrics(labels, values, threshold)})
                        runtimes.append({"split": "test", "training_seed": int(payload["seed"]),
                                         "primary_method": payload["primary_method"], **identity,
                                         "inference_seconds": elapsed,
                                         "feature_seconds": meta["physics_feature_seconds"] + meta["context_feature_seconds"],
                                         "events_per_second": len(labels) / max(elapsed + meta["physics_feature_seconds"] + meta["context_feature_seconds"], EPS)})
                    print(f"evaluated context-v2 {scene}, f={fraction:g}, {kind} {rate:g}, seed={base_seed}", flush=True)
    write_csv(RUN / "event_metrics.csv", rows)
    write_csv(RUN / "runtime_event.csv", runtimes)
    write_json(RUN / "test_case_manifest.json", cases)
    manifest["test_evaluation_started"] = True
    manifest["test_evaluation_completed"] = True
    write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "probe", "train", "evaluate", "all"))
    args = parser.parse_args()
    c = cfg()
    RUN.mkdir(parents=True, exist_ok=True)
    freeze_protocol(c)
    if args.action in {"prepare", "all"}:
        prepare_split("train", c)
        prepare_split("validation", c)
    if args.action == "probe":
        run_probe(c)
    if args.action in {"train", "all"}:
        run_train(c)
    if args.action in {"evaluate", "all"}:
        run_evaluate(c)


if __name__ == "__main__":
    main()
