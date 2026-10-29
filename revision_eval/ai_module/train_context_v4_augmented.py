"""Train v4 with corrected measurement masks and synthetic auxiliary labels.

The v4 branch is independent of v2/v3.  A small translated-event set is used
only to regularize the physics-context expert: its source/noise labels are
appended to that branch's training minibatches, while the real cached records
remain the dominant data source.  The displacement loss uses complete phase
windows, including injected noise, so its effect is measurable rather than a
source-only consistency check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import train_context_v2 as v2
import train_context_v3 as base


HERE = Path(__file__).resolve().parent
base.PROTOCOL_PATH = HERE / "context_v4_augmented_protocol.json"
base.RUN = HERE / "runs" / "context_v4_augmented"
base.SCHEMA = "vgs_physics_ai_contextual_reliability_v4_augmented"
base.MEASUREMENT_WEIGHT = 0.20


def synthetic_measurement_cases(joint_norm, count=24, seed=12031):
    """Generate translated pairs; masks include all events in each phase."""
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
        source = np.concatenate([
            np.ones(n_signal * 3, dtype=np.float32),
            np.zeros(n_noise, dtype=np.float32),
        ])
        order = np.argsort(events[:, 0], kind="mergesort")
        events, source = events[order], source[order]
        physics, _ = v2.extract_physics(events, 64, 64)
        context = v2.extract_context(events, 64, 64)
        joint = np.concatenate((physics, context), axis=1)
        phase1 = (events[:, 0] >= 100_000) & (events[:, 0] < 102_000)
        phase2 = (events[:, 0] >= 110_000) & (events[:, 0] < 112_000)
        if not phase1.any() or not phase2.any():
            raise RuntimeError("synthetic phase unexpectedly empty")
        cases.append({
            "features": v2.standardize(joint, joint_norm),
            "labels": source,
            "phase1": phase1,
            "phase2": phase2,
            "coordinates": events[:, 1:3].astype(np.float32),
            "target": np.asarray([float(dx), float(dy)], dtype=np.float32),
        })
    return cases


base.synthetic_measurement_cases = synthetic_measurement_cases
_original_train_reliability = base.train_reliability


def train_reliability(model, x_train, labels, validation_cases, c, seed, label,
                      measurement_cases=None, epochs=None):
    """Append known synthetic labels only to the physics-context branch."""
    if measurement_cases is not None and label == "physics_context_mlp":
        aux_x = np.concatenate([case["features"] for case in measurement_cases], axis=0)
        aux_y = np.concatenate([case["labels"] for case in measurement_cases], axis=0)
        x_train = np.concatenate((x_train, aux_x), axis=0).astype(np.float32, copy=False)
        labels = np.concatenate((labels, aux_y), axis=0).astype(np.float32, copy=False)
    return _original_train_reliability(
        model, x_train, labels, validation_cases, c, seed, label,
        measurement_cases=measurement_cases, epochs=epochs,
    )


base.train_reliability = train_reliability


if __name__ == "__main__":
    base.main()
