"""Select original event rows with a verified bundled v4 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision_eval/ai_module"))


def checkpoint_for_seed(seed):
    records = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))["models"]
    record = next(r for r in records if r["seed"] == seed and r["variant"] == "context_v4_augmented")
    path = (ROOT / record["path"]).resolve()
    if ROOT not in path.parents:
        raise ValueError("Checkpoint must be inside this repository")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Checkpoint hash does not match the bundled manifest")
    return path


def validate(events, height, width):
    if height <= 0 or width <= 0:
        raise ValueError("Sensor dimensions must be positive")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Expected an N x 4 array with columns (t_us, x, y, p)")
    if not np.issubdtype(events.dtype, np.integer):
        raise ValueError("Input must use integer microseconds and integer event fields")
    if events.size and (events.min() < 0 or events.max() > np.iinfo(np.int64).max):
        raise ValueError("Event fields must be nonnegative and fit signed int64")
    events = events.astype(np.int64, copy=False)
    if len(events):
        if np.any(events[1:, 0] < events[:-1, 0]):
            raise ValueError("Input must already be sorted by timestamp")
        if np.any(events[:, 1] >= width) or np.any(events[:, 2] >= height):
            raise ValueError("Coordinates exceed the declared sensor dimensions")
        if not np.isin(events[:, 3], [0, 1]).all():
            raise ValueError("Polarity must be encoded as 0 or 1")
    return events


def select(events, height, width, seed=6101):
    events = validate(np.asarray(events), height, width)
    checkpoint = checkpoint_for_seed(int(seed))
    if len(events) == 0:
        return events.copy(), np.empty(0, dtype=bool), {"seed": seed, "input_events": 0, "retained_events": 0, "fixed_latency_us": 5000}
    from downstream_ebiv_context_v2 import infer_context_v2
    result = infer_context_v2(events, height, width, checkpoint)
    keep = result["keep"]
    metadata = {"seed": int(seed), "input_events": len(events), "retained_events": int(keep.sum()),
                "threshold": result["threshold"], "checkpoint_sha256": result["checkpoint_sha256"], **result["metadata"]}
    return events[keep], keep, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=[6101, 6102, 6103], default=6101)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy":
        parser.error("--output must end in .npy")
    targets = [args.output, args.output.with_suffix(".keep.npy"), args.output.with_suffix(".json")]
    if any(p.exists() for p in targets):
        parser.error("An output already exists; choose a new output filename")
    events = np.load(args.input, allow_pickle=False)
    selected, keep, metadata = select(events, args.height, args.width, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(targets[0], selected, allow_pickle=False)
    np.save(targets[1], keep, allow_pickle=False)
    targets[2].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
