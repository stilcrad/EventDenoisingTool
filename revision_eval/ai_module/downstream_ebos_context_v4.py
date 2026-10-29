"""Run the frozen v4 reliability network on the fixed-ROI EBOS protocol."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import downstream_ebos_gated as benchmark
from downstream_ebiv_context_v2 import infer_context_v2


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "runs" / "context_v4_augmented" / "ebos"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = benchmark.build_parser()
    parser.description = "Run frozen context-v4 event selection with the fixed-ROI EBOS protocol."
    parser.set_defaults(
        out_dir=DEFAULT_OUT,
        learned_method_prefix="context_v4",
        learned_method_label="Physics-context reliability MLP v4 (frozen)",
        learned_method_kind="context_v4_primary",
        result_stem="ebos_context_v4",
        protocol_name="frozen_context_v4_augmented_on_original_ebos_config_roi_and_time_window",
        selection_state=(
            "v4 asymmetric source-protection, sparsity augmentation, synthetic reliability labels and corrected "
            "measurement-auxiliary training; validation-only threshold; raw event output"
        ),
    )
    parser.add_argument(
        "--context-method",
        choices=("context_ai_expert", "physics_context_mlp", "gated_context_dual_expert", "physics_calibrated"),
        default=None,
        help="evaluate a frozen ablation branch; default uses the validation-selected primary",
    )
    parser.add_argument("--context-threshold", type=float, default=None,
                        help="optional explicit threshold for diagnostics")
    args = parser.parse_args(argv)
    benchmark.infer_gated_from_checkpoint = partial(
        infer_context_v2, method=args.context_method, threshold_override=args.context_threshold,
    )
    args.repo = benchmark._absolute_path(args.repo)
    args.package = benchmark._absolute_path(args.package)
    args.out_dir = benchmark._absolute_path(args.out_dir)
    args.python = benchmark._absolute_path(args.python)
    if len(args.scene) > 1 and any(value is not None for value in (args.config, args.raw_hdf, args.clean_hdf, args.reference_dir)):
        raise ValueError("--config/--raw-hdf/--clean-hdf/--reference-dir require exactly one --scene")
    if not args.repo.is_dir():
        raise FileNotFoundError(f"event_based_bos-main repository not found: {args.repo}")
    if not args.package.is_dir() and any(value is None for value in (args.raw_hdf, args.clean_hdf)):
        raise FileNotFoundError(f"historical EBOS package not found: {args.package}")
    if len(set(args.checkpoint)) != len(args.checkpoint):
        raise ValueError("each frozen checkpoint must appear once")
    event_rows, flow_rows, scene_metadata = [], [], []
    for scene in args.scene:
        event_part, flow_part, metadata = benchmark.run_scene(args, scene)
        event_rows.extend(event_part)
        flow_rows.extend(flow_part)
        scene_metadata.append(metadata)
        print(f"prepared {scene}", flush=True)
    benchmark._save_csv(args.out_dir / f"{args.result_stem}_event_metrics.csv", event_rows)
    benchmark._save_csv(args.out_dir / f"{args.result_stem}_flow_frames.csv", flow_rows)
    benchmark._save_csv(args.out_dir / f"{args.result_stem}_flow_summary.csv",
                        [row for scene in scene_metadata for row in scene["rows"]])
    metadata = {
        "protocol": str(args.protocol_name),
        "status": "completed" if args.run_solver else "prepared_no_solver",
        "repo": str(args.repo), "package": str(args.package), "out_dir": str(args.out_dir),
        "checkpoints": [{"path": str(benchmark._absolute_path(path)), "sha256": benchmark.sha256_file(path)}
                         for path in args.checkpoint],
        "reference": "clean-source EBOS solver output; source-derived reference, not absolute physical ground truth",
        "latency_and_format": "fixed 5 ms lookahead; timestamp-aligned subset of original events; no frame accumulation",
        "scenes": scene_metadata,
    }
    benchmark._dump_json(args.out_dir / f"{args.result_stem}_metadata.json", metadata)
    print(json.dumps({"status": metadata["status"], "out_dir": str(args.out_dir),
                      "scenes": len(args.scene)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
