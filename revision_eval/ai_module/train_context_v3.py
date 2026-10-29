"""Train the physics-aware event-reliability network (context-v3).

This script reuses the frozen context-v2 caches but changes the learning
objective.  The small MLP is trained with an asymmetric false-rejection cost,
deterministic sparsity augmentation, and a differentiable displacement
regularizer on synthetic translated event pairs with known motion.  No
held-out hot-plate data are read during training or threshold selection.
"""
from __future__ import annotations

import copy
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
import torch.nn.functional as F

import train_context_v2 as v2


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "context_v3_protocol.json"
RUN = HERE / "runs" / "context_v3"
SCHEMA = "vgs_physics_ai_contextual_reliability_v3"
POSITIVE_WEIGHT = 3.0
MEASUREMENT_WEIGHT = 0.05
SPARSITY_PHYSICS = 0.05
SPARSITY_CONTEXT = 0.10


def protocol():
    value = v2.read_json(PROTOCOL_PATH)
    if value.get("status") != "independent training protocol after context-v2 downstream audit":
        raise RuntimeError("context-v3 protocol is not frozen")
    return value


def sparse_input(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Replace a small random fraction of standardized descriptors by their mean."""
    result = x.clone()
    if result.shape[1] >= 14:
        physics_keep = torch.rand((len(result), 14), generator=generator) >= SPARSITY_PHYSICS
        result[:, :14] = result[:, :14] * physics_keep.to(result.dtype)
    if result.shape[1] > 14:
        context_keep = torch.rand((len(result), result.shape[1] - 14), generator=generator) >= SPARSITY_CONTEXT
        result[:, 14:] = result[:, 14:] * context_keep.to(result.dtype)
    return result


def synthetic_measurement_cases(joint_norm, count=24, seed=12031):
    """Generate small translated event pairs and their known displacement targets."""
    rng = np.random.default_rng(int(seed))
    cases = []
    for _ in range(int(count)):
        n_signal = 280
        n_noise = 160
        dx = int(rng.integers(-3, 4))
        dy = int(rng.integers(-3, 4))
        if dx == 0 and dy == 0:
            dx = 1
        x1 = rng.integers(8, 56, n_signal, dtype=np.int64)
        y1 = rng.integers(8, 56, n_signal, dtype=np.int64)
        x2 = np.clip(x1 + dx, 2, 61)
        y2 = np.clip(y1 + dy, 2, 61)
        t1 = 100_000 + rng.integers(0, 2_000, n_signal, dtype=np.int64)
        t2 = 110_000 + rng.integers(0, 2_000, n_signal, dtype=np.int64)
        # A third burst provides genuine future context for the second burst.
        xt = np.clip(x2 + rng.integers(-1, 2, n_signal), 2, 61)
        yt = np.clip(y2 + rng.integers(-1, 2, n_signal), 2, 61)
        tt = 120_000 + rng.integers(0, 2_000, n_signal, dtype=np.int64)
        tn = rng.integers(100_000, 122_000, n_noise, dtype=np.int64)
        xn = rng.integers(2, 62, n_noise, dtype=np.int64)
        yn = rng.integers(2, 62, n_noise, dtype=np.int64)
        p1 = rng.integers(0, 2, n_signal, dtype=np.int64)
        p2 = rng.integers(0, 2, n_signal, dtype=np.int64)
        pt = rng.integers(0, 2, n_signal, dtype=np.int64)
        pn = rng.integers(0, 2, n_noise, dtype=np.int64)
        events = np.concatenate([
            np.stack((t1, x1, y1, p1), axis=1),
            np.stack((t2, x2, y2, p2), axis=1),
            np.stack((tt, xt, yt, pt), axis=1),
            np.stack((tn, xn, yn, pn), axis=1),
        ], axis=0)
        phase1 = np.concatenate([
            np.ones(n_signal, dtype=bool), np.zeros(n_signal + n_signal + n_noise, dtype=bool)
        ])
        phase2 = np.concatenate([
            np.zeros(n_signal, dtype=bool), np.ones(n_signal, dtype=bool),
            np.zeros(n_signal + n_noise, dtype=bool)
        ])
        order = np.argsort(events[:, 0], kind="mergesort")
        events = events[order]
        phase1, phase2 = phase1[order], phase2[order]
        physics, _ = v2.extract_physics(events, 64, 64)
        context = v2.extract_context(events, 64, 64)
        joint = np.concatenate((physics, context), axis=1)
        cases.append({
            "features": v2.standardize(joint, joint_norm),
            "phase1": phase1,
            "phase2": phase2,
            "coordinates": events[:, 1:3].astype(np.float32),
            "target": np.asarray([float(dx), float(dy)], dtype=np.float32),
        })
    return cases


def measurement_loss(model: nn.Module, cases) -> torch.Tensor:
    losses = []
    for case in cases:
        x = torch.from_numpy(case["features"])
        p = torch.sigmoid(model(x)).reshape(-1).clamp(1.0e-4, 1.0)
        coords = torch.from_numpy(case["coordinates"])
        m1 = torch.from_numpy(case["phase1"])
        m2 = torch.from_numpy(case["phase2"])
        w1, w2 = p[m1], p[m2]
        c1 = (w1[:, None] * coords[m1]).sum(dim=0) / w1.sum().clamp_min(1.0e-4)
        c2 = (w2[:, None] * coords[m2]).sum(dim=0) / w2.sum().clamp_min(1.0e-4)
        target = torch.from_numpy(case["target"])
        losses.append(F.smooth_l1_loss(c2 - c1, target))
    return torch.stack(losses).mean() if losses else torch.zeros((), requires_grad=True)


def train_reliability(model, x_train, labels, validation_cases, c, seed, label,
                      measurement_cases=None, epochs=None):
    v2.set_cpu_seed(int(seed))
    torch.set_num_threads(4)
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(labels.astype(np.float32))[:, None]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(c["optimization"]["learning_rate"]),
        weight_decay=float(c["optimization"]["weight_decay"]),
    )
    epochs = int(epochs or c["optimization"]["epochs"])
    best_auc, best_state, best_epoch = -math.inf, None, 0
    history = []
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(int(seed) + 17_003)
    for epoch in range(epochs):
        order = torch.randperm(len(x_tensor), generator=generator)
        total = 0.0
        model.train()
        for start in range(0, len(order), int(c["optimization"]["batch_size"])):
            index = order[start:start + int(c["optimization"]["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            xb = sparse_input(x_tensor[index], generator)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, y_tensor[index], reduction="none")
            weights = torch.where(y_tensor[index] > 0.5, POSITIVE_WEIGHT, 1.0)
            loss = (bce * weights).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        aux_value = 0.0
        if measurement_cases:
            optimizer.zero_grad(set_to_none=True)
            aux = measurement_loss(model, measurement_cases)
            (MEASUREMENT_WEIGHT * aux).backward()
            optimizer.step()
            aux_value = float(aux.detach())
        score = v2.macro_auc(model, validation_cases)
        history.append({
            "epoch": epoch + 1,
            "train_weighted_bce": total / len(order),
            "train_measurement_loss": aux_value,
            "macro_validation_auc": score,
        })
        if score > best_auc:
            best_auc, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    print(f"trained {label}_{seed}: validation macro AUC={best_auc:.5f}, epoch={best_epoch}", flush=True)
    return model, {
        "best_epoch": best_epoch,
        "macro_validation_auc": best_auc,
        "training_seconds": time.perf_counter() - started,
        "positive_weight": POSITIVE_WEIGHT,
        "measurement_weight": MEASUREMENT_WEIGHT if measurement_cases else 0.0,
        "sparsity_physics": SPARSITY_PHYSICS,
        "sparsity_context": SPARSITY_CONTEXT,
        "history": history,
    }


def measurement_report(model, cases):
    before, after = [], []
    with torch.no_grad():
        for case in cases:
            coords = case["coordinates"]
            m1, m2 = case["phase1"], case["phase2"]
            raw_delta = coords[m2].mean(axis=0) - coords[m1].mean(axis=0)
            p = torch.sigmoid(model(torch.from_numpy(case["features"]))).reshape(-1).numpy()
            c1 = (p[m1, None] * coords[m1]).sum(axis=0) / max(float(p[m1].sum()), 1.0e-4)
            c2 = (p[m2, None] * coords[m2]).sum(axis=0) / max(float(p[m2].sum()), 1.0e-4)
            target = case["target"]
            before.append(float(np.linalg.norm(raw_delta - target)))
            after.append(float(np.linalg.norm((c2 - c1) - target)))
    return {
        "cases": len(cases),
        "raw_centroid_error_mean_px": float(np.mean(before)),
        "weighted_centroid_error_mean_px": float(np.mean(after)),
        "raw_centroid_error_sd_px": float(np.std(before, ddof=1)),
        "weighted_centroid_error_sd_px": float(np.std(after, ddof=1)),
    }


def save_checkpoint(seed, result, c, aux_report):
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "models" / f"context_v3_{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "seed": int(seed),
        "primary_method": result["primary_method"],
        "selection_scores": result["selection_scores"],
        "thresholds": result["thresholds"],
        "threshold_info": result["threshold_info"],
        "context_ai_state": result["context_model"].state_dict(),
        "physics_context_state": result["joint_model"].state_dict(),
        "gate_state": result["gate"].state_dict(),
        "physics_calibration": {"eps": v2.EPS},
        "physics_calibration_state": result["calibrator"].state_dict(),
        "context_normalization": {"mean": result["context_norm"][0], "scale": result["context_norm"][1]},
        "physics_normalization": {"mean": result["physics_norm"][0], "scale": result["physics_norm"][1]},
        "joint_normalization": {"mean": result["joint_norm"][0], "scale": result["joint_norm"][1]},
        "fixed_latency_us": int(v2.LOOKAHEAD_US),
        "training_protocol": str(PROTOCOL_PATH.resolve()),
        "training_protocol_sha256": v2.file_hash(PROTOCOL_PATH),
        "measurement_auxiliary_report": aux_report,
        "training": result["training"],
    }
    torch.save(payload, path)
    return path


def build(seed, c, train_records, validation_records):
    physics, context, scores, labels = v2.concatenate(train_records)
    context_norm, physics_norm = v2.norm_fit(context), v2.norm_fit(physics)
    joint = np.concatenate((physics, context), axis=1)
    joint_norm = v2.norm_fit(joint)
    x_context = v2.standardize(context, context_norm)
    x_joint = v2.standardize(joint, joint_norm)
    context_cases, joint_cases, raw_validation = v2.selection_cases(validation_records, context_norm, joint_norm)
    calibrator = v2.fit_platt(scores, labels, class_balanced=True, max_iter=200, seed=7002).freeze()
    aux_cases = synthetic_measurement_cases(joint_norm)
    v2.set_cpu_seed(int(seed))
    context_model, context_info = train_reliability(
        v2.TinyMLP(100), x_context, labels, context_cases, c, seed,
        "context_ai_expert", measurement_cases=None,
    )
    v2.set_cpu_seed(int(seed) + 1)
    joint_model, joint_info = train_reliability(
        v2.TinyMLP(114), x_joint, labels, joint_cases, c, seed,
        "physics_context_mlp", measurement_cases=aux_cases,
    )
    aux_report = measurement_report(joint_model, aux_cases)
    x_physics = v2.standardize(physics, physics_norm)
    p_physics = v2.calibrated_probabilities(calibrator, scores)
    p_ai = v2.probabilities(context_model, x_context)
    x_gate = v2.gate_input(x_physics, x_context, p_physics, p_ai)
    gate_validation = []
    for vp, vc, vs, vy in raw_validation:
        vps, vcs = v2.standardize(vp, physics_norm), v2.standardize(vc, context_norm)
        vpp, vpa = v2.calibrated_probabilities(calibrator, vs), v2.probabilities(context_model, vcs)
        gate_validation.append((v2.gate_input(vps, vcs, vpp, vpa), vpp, vpa, vy))
    v2.set_cpu_seed(int(seed) + 2)
    gate, gate_info = v2.train_gate(
        v2.TinyMLP(117, hidden=(64, 16)), x_gate, labels, p_physics, p_ai,
        gate_validation, c, seed,
    )
    validation_predictions = {"context_ai_expert": [], "physics_context_mlp": [],
                              "gated_context_dual_expert": [], "physics_calibrated": []}
    validation_labels = []
    validation_rows = []
    for record in validation_records:
        vp, vc, vs, vy, _ = v2.load_cache(record)
        vps, vcs = v2.standardize(vp, physics_norm), v2.standardize(vc, context_norm)
        vjoint = v2.standardize(np.concatenate((vp, vc), axis=1), joint_norm)
        vpp = v2.calibrated_probabilities(calibrator, vs)
        vpa = v2.probabilities(context_model, vcs)
        predictions = {
            "context_ai_expert": vpa,
            "physics_context_mlp": v2.probabilities(joint_model, vjoint),
            "gated_context_dual_expert": v2.gated_probabilities(
                gate, v2.gate_input(vps, vcs, vpp, vpa), vpp, vpa
            ),
            "physics_calibrated": vpp,
        }
        validation_labels.append(vy)
        for name, value in predictions.items():
            validation_predictions[name].append(value)
    thresholds, threshold_info = {}, {}
    for name, values in validation_predictions.items():
        thresholds[name], threshold_info[name] = v2.operating_threshold(values, validation_labels)
    selection_scores = {
        "context_ai_expert": float(context_info["macro_validation_auc"]),
        "physics_context_mlp": float(joint_info["macro_validation_auc"]),
        "gated_context_dual_expert": float(gate_info["macro_validation_auc"]),
    }
    tie_order = {"gated_context_dual_expert": 2, "physics_context_mlp": 1, "context_ai_expert": 0}
    primary = max(selection_scores, key=lambda name: (selection_scores[name], tie_order[name]))
    for record, vy, predictions in zip(
        validation_records,
        validation_labels,
        zip(*[validation_predictions[name] for name in validation_predictions]),
    ):
        identity = {key: record[key] for key in (
            "scene", "fraction", "noise_type", "rate", "base_noise_seed", "effective_noise_seed"
        )}
        for name, value in zip(validation_predictions, predictions):
            validation_rows.append({
                "split": "validation",
                "training_seed": int(seed),
                "method": name,
                "threshold": thresholds[name],
                **identity,
                **v2.metrics(vy, value, thresholds[name]),
            })
    return {
        "seed": int(seed),
        "primary_method": primary,
        "selection_scores": selection_scores,
        "thresholds": thresholds,
        "threshold_info": threshold_info,
        "training": {
            "context_ai_expert": context_info,
            "physics_context_mlp": joint_info,
            "gated_context_dual_expert": gate_info,
        },
        "calibrator": calibrator,
        "context_model": context_model,
        "joint_model": joint_model,
        "gate": gate,
        "context_norm": context_norm,
        "physics_norm": physics_norm,
        "joint_norm": joint_norm,
        "validation_rows": validation_rows,
        "measurement_auxiliary_report": aux_report,
    }


def main() -> None:
    c = protocol()
    parent = v2.read_json(v2.PROTOCOL_PATH)
    train_records, validation_records = v2.cache_records("train"), v2.cache_records("validation")
    if len(train_records) != 30 or len(validation_records) != 24:
        raise RuntimeError("unexpected context-v2 cache counts")
    RUN.mkdir(parents=True, exist_ok=True)
    v2.write_json(RUN / "protocol_snapshot.json", c)
    v2.write_json(RUN / "parent_protocol_snapshot.json", parent)
    all_rows, manifest = [], []
    for seed in c["training_seeds"]:
        result = build(int(seed), parent, train_records, validation_records)
        path = save_checkpoint(int(seed), result, parent, result["measurement_auxiliary_report"])
        all_rows.extend(result["validation_rows"])
        manifest.append({
            "seed": int(seed),
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": v2.file_hash(path),
            "schema": SCHEMA,
            "primary_method": result["primary_method"],
            "selection_scores": result["selection_scores"],
            "thresholds": result["thresholds"],
            "threshold_info": result["threshold_info"],
            "measurement_auxiliary_report": result["measurement_auxiliary_report"],
        })
        print(f"frozen context-v3 checkpoint {seed}; primary={result['primary_method']}", flush=True)
    v2.write_csv(RUN / "validation_metrics.csv", all_rows)
    v2.write_json(RUN / "frozen_manifest.json", {
        "schema": SCHEMA,
        "protocol_sha256": v2.file_hash(PROTOCOL_PATH),
        "parent_protocol_sha256": v2.file_hash(v2.PROTOCOL_PATH),
        "models": manifest,
        "selection_data": "context-v2 train/validation caches and synthetic measurement auxiliary only",
        "test_data_used_for_selection": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": "cpu"},
    })
    print(RUN / "frozen_manifest.json")


if __name__ == "__main__":
    main()
