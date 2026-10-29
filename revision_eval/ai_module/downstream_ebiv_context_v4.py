"""Run the frozen v4 reliability network on the original EBIV protocol."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import downstream_ebiv_gated as benchmark
from downstream_ebiv_context_v2 import infer_context_v2


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "runs" / "context_v4_augmented" / "ebiv_lookahead_corrected"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = benchmark.build_parser()
    parser.description = "Run frozen context-v4 AI selection on the original EBIV noise benchmark."
    parser.set_defaults(
        out_dir=DEFAULT_OUT,
        feature_lookahead_us=5000,
        learned_method_prefix="context_v4",
        learned_method_kind="context_v4_primary",
        result_stem="ebiv_context_v4_lookahead_corrected",
        protocol_name="original_ebiv_uniform_noise_with_frozen_context_v4_primary_and_real_5ms_future_context",
        selection_note=(
            "physics and context state use the raw chronological stream; learned features include real events through "
            "active_end + 5 ms; only active-window events are selected; recording-tail events remain unchanged"
        ),
        report_title="Frozen context-v4 AI EBIV benchmark (corrected lookahead)",
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
    result = benchmark.run_benchmark(args)
    print(json.dumps({"status": result["status"], "out_dir": str(Path(args.out_dir).resolve()),
                      "n_rows": len(result["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
