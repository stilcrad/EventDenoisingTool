"""Interface tests, not a numerical reproduction of the paper."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from denoise_npy import select, validate, checkpoint_for_seed


def main():
    events = np.load(ROOT / "data/synthetic_example/events.npy", allow_pickle=False)
    for seed in [6101, 6102, 6103]:
        selected, keep, meta = select(events, 24, 32, seed)
        assert keep.shape == (len(events),) and keep.dtype == bool
        np.testing.assert_array_equal(selected, events[keep])
        assert keep[events[:, 0] > int(events[-1, 0]) - 5000].all()
        assert meta["inference_method"] == "physics_context_mlp"
        assert meta["fixed_latency_us"] == 5000
        assert 0 <= meta["threshold"] <= 1
    empty, mask, _ = select(np.empty((0, 4), dtype=np.int64), 24, 32)
    assert empty.shape == (0, 4) and mask.shape == (0,)
    for invalid in [events.astype(float), events[::-1], np.array([[1, 32, 0, 0]]), np.array([[1, 0, 0, -1]])]:
        try:
            validate(invalid, 24, 32)
        except ValueError:
            continue
        raise AssertionError("Invalid input was accepted")
    from train_context_v2 import restore
    for variant in ["context_v4_augmented", "v4_no_measurement"]:
        for seed in [6101, 6102, 6103]:
            path = ROOT / "revision_eval/ai_module/runs" / variant / "models" / f"context_v3_{seed}.pt"
            payload, *_ = restore(path)
            assert payload["seed"] == seed and payload["fixed_latency_us"] == 5000
    print("PASS: six checkpoints load; three-seed event selection; event preservation; recording-tail handling; invalid-input checks")


if __name__ == "__main__":
    main()
