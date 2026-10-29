# Checkpoints

The six checkpoint files remain under their original relative `revision_eval/ai_module/runs/<variant>/models` directories so the independent synthetic evaluation can locate both variants.

`manifest.json` records release hashes and archived hashes. Re-serialization changes file hashes. The only changed payload field is `training_protocol`, now repository-relative. Every other field was compared recursively, including all tensors, arrays, frozen thresholds and training summaries.

These files use PyTorch's pickle-based format. Load only the bundled files after checking their recorded hashes. Do not load an arbitrary checkpoint supplied by a third party. The manifest establishes consistency with this package, not cryptographic authenticity of an unknown download.

Model redistribution and its license still require author approval.
