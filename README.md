# VGS event reliability learning

Local release candidate for **Physics-Guided Event Reliability Learning for Event-Based Photomechanics**.

Paper DOI: https://doi.org/10.1007/s10409-026-60785-x

This directory stages the context-v4 learning branch for release review. It is not yet a public, licensed software release. See [RELEASE_STATUS.md](RELEASE_STATUS.md) before publishing.

## Method

The direct-fusion network combines 14 physical descriptors with 100 event-centred time-surface descriptors. It predicts event reliability and selects a subset of the original events. Coordinates, timestamps and polarities are unchanged.

Features use 5 ms of future context. Events in the final 5 ms of a finite recording are retained unchanged. The method is not a zero-delay causal filter. Source events and injected noise provide proxy labels. Source-stream measurement references do not establish absolute accuracy against independent physical ground truth.

The training objective combines asymmetric event classification with an auxiliary weighted-centroid displacement loss. The dual-expert gate is retained as a comparison branch. The default inference method is the frozen validation-selected direct-fusion model.

## Contents

| Directory | Contents |
| --- | --- |
| `revision_eval/ai_module` | Training, inference, feature extraction and benchmark wrappers |
| `revision_eval/scripts` | Analytical VGS reference |
| `revision_eval/ai_module/runs/*/models` | Three full-model and three zero-measurement-loss checkpoints |
| `models` | Checkpoint hashes and provenance |
| `data` | Data instructions and an artificial interface-test example |
| `results` | Unmodified archived CSV files that contain no local machine paths |
| `scripts` | NumPy inference entry point and release checks |
| `tests` | Release smoke tests |

The inherited `v2` and `v3` filenames are shared implementations used by the v4 trainer. A checkpoint named `context_v3_6101.pt` can have a v4 schema. Use the schema and manifest, not its filename alone, to identify a model.

## Environment

Use Python 3.9 or later and install the packages in `requirements.txt` in a dedicated environment. These requirements are not an independently validated version lock. CPU PyTorch is sufficient for the supplied inference checks.

```bash
python -m pip install -r requirements.txt
python scripts/check_release.py
python tests/test_release_smoke.py
```

## Run event selection

Input is a chronological, integer NumPy array of shape `(N, 4)` with columns `(t, x, y, p)`. Timestamps are in microseconds. Polarity is encoded as 0 or 1. Sensor dimensions must match the coordinates.

```bash
python scripts/denoise_npy.py --input data/synthetic_example/events.npy --height 24 --width 32 --output outputs/example_denoised.npy --seed 6101
```

The example is generated artificial data for interface testing. It is not an experimental measurement and cannot reproduce a paper table. The script validates the bundled checkpoint hash before loading it. Do not load untrusted PyTorch pickle checkpoints.

## Train and evaluate

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the original training entry points, data layout and external dependencies. Complete real-data retraining and EBOS/EBIV reproduction have not been repeated for this release directory. This package does not yet reproduce every legacy table in the manuscript.

Archived results include unfavorable outcomes. Measurement-loss ablations and independent synthetic cases are distinct experiments. Low-light and high-noise limitations must be retained in any comparison.

## Publication and data

No data repository DOI or public data URL has been assigned to this local package. The paper DOI is not a dataset identifier. Code availability does not replace the journal's data-availability requirement. See [data/README.md](data/README.md).

Licensing and redistribution permissions remain to be confirmed by the authors. No open-source license is asserted by this preparation step.
