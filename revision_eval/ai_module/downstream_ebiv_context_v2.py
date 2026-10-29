"""Run frozen context-v2 checkpoints in the original EBIV benchmark.

The original EBIV noise generator, historical temporal-support VGS baseline,
solver, source-derived velocity reference, and coverage accounting are reused
from ``downstream_ebiv_gated.py``.  Only the learned inference callback is
replaced.  Events in the final 5 ms of the finite recording are right-censored
for the declared lookahead and therefore retained unchanged.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

import downstream_ebiv_gated as benchmark
from features import extract as extract_physics
from features_context import LOOKAHEAD_US, complete_context_mask, extract_context
from train_context_v2 import predict_all, restore


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "runs" / "context_v2" / "ebiv"


def infer_context_v2(
    events: np.ndarray,
    height: int,
    width: int,
    checkpoint: Union[str, Path, Mapping],
    *,
    batch_size: int = 65_536,
    return_features: bool = False,
    method: Optional[str] = None,
    threshold_override: Optional[float] = None,
):
    if isinstance(checkpoint, Mapping):
        raise TypeError("context-v2 EBIV inference requires a frozen checkpoint path for provenance")
    path = Path(checkpoint).expanduser().resolve()
    events = np.asarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("expected chronological N x 4 (t,x,y,p) events")
    if len(events) and np.any(events[1:, 0] < events[:-1, 0]):
        raise ValueError("events must be time sorted")
    payload, *_ = restored = restore(path)
    if int(payload.get("fixed_latency_us", -1)) != int(LOOKAHEAD_US):
        raise ValueError("checkpoint latency differs from context feature implementation")
    physics, scores = extract_physics(events, int(height), int(width))
    context = extract_context(events, int(height), int(width))
    predictions = predict_all(restored, physics, context, scores)
    checkpoint_primary = str(payload["primary_method"])
    selected_method = checkpoint_primary if method is None else str(method)
    if selected_method not in predictions:
        raise KeyError(f"checkpoint method {selected_method!r} is unavailable")
    threshold = (
        float(payload["thresholds"][selected_method])
        if threshold_override is None
        else float(threshold_override)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"selection threshold must be in [0, 1], got {threshold}")
    keep = np.asarray(predictions[selected_method] >= threshold, dtype=bool)
    complete = complete_context_mask(events[:, 0])
    # A finite recording provides no future context for its final 5 ms.  Those
    # events are passed through instead of being assigned artificial context.
    keep[~complete] = True
    result = {
        "keep": keep,
        "alpha": None,
        "p_physics": np.asarray(predictions["physics_calibrated"], np.float32),
        "p_ai": np.asarray(predictions["context_ai_expert"], np.float32),
        "probability": np.asarray(predictions[selected_method], np.float32),
        "threshold": threshold,
        "checkpoint_path": str(path),
        "checkpoint_sha256": benchmark.sha256_file(path),
        "checkpoint_schema": payload["schema"],
        "training_seed": int(payload["seed"]),
        "metadata": {
            "checkpoint_primary_method": checkpoint_primary,
            "inference_method": selected_method,
            "threshold_source": "checkpoint_validation" if threshold_override is None else "explicit_override",
            "fixed_latency_us": int(LOOKAHEAD_US),
            "complete_context_events": int(complete.sum()),
            "right_censored_events": int((~complete).sum()),
            "right_censored_policy": "retain unchanged",
            "event_format": "timestamp-aligned original-event subset; no frame accumulation",
        },
    }
    if return_features:
        result["physics_features"] = physics
        result["context_features"] = context
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = benchmark.build_parser()
    parser.description = "Run frozen context-v2 AI selection on the original EBIV noise benchmark."
    parser.set_defaults(
        out_dir=DEFAULT_OUT,
        feature_lookahead_us=LOOKAHEAD_US,
        learned_method_prefix="context_v2",
        learned_method_kind="context_v2_primary",
        result_stem="ebiv_context_v2",
        protocol_name="original_ebiv_uniform_noise_with_frozen_context_v2_primary",
        selection_note=(
            "physics state uses the raw chronological stream; context uses real events through active_end + 5 ms; "
            "only active-window events are selected, and genuinely right-censored recording-tail events are retained"
        ),
        report_title="Frozen context-v2 AI EBIV benchmark",
    )
    parser.add_argument(
        "--context-method",
        choices=("context_ai_expert", "physics_context_mlp", "gated_context_dual_expert", "physics_calibrated"),
        default=None,
        help="evaluate a frozen ablation branch; default uses the validation-selected checkpoint primary",
    )
    parser.add_argument(
        "--context-threshold",
        type=float,
        default=None,
        help="optional explicit threshold for diagnostic sweeps; omit for the frozen validation threshold",
    )
    args = parser.parse_args(argv)
    benchmark.infer_gated_from_checkpoint = partial(
        infer_context_v2,
        method=args.context_method,
        threshold_override=args.context_threshold,
    )
    result = benchmark.run_benchmark(args)
    print(json.dumps({"status": result["status"], "out_dir": str(Path(args.out_dir).resolve()),
                      "n_rows": len(result["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
