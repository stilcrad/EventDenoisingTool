# Reproduction notes

## Frozen inference

Run `scripts/denoise_npy.py` with the bundled checkpoint selected by seed. The wrapper checks integer timestamps, ordering, coordinates and 0/1 polarity. It returns the original selected event rows. The final 5 ms of an observed recording are right-censored and pass through unchanged.

## Training

Obtain the authorized CCS recordings and configure `VGS_DATA_ROOT` as described in `data/README.md`. The cache preparation and training entry points are inherited from the archived implementation.

```bash
python revision_eval/ai_module/train_context_v2.py prepare
python revision_eval/ai_module/train_context_v4_augmented.py
```

These commands require actual recordings and substantial compute. They have not been run as part of packaging. The first command prepares the shared train/validation feature caches. The second applies the v4 asymmetric classification and measurement objective. Existing checkpoint files are written by the trainer. Back up the frozen release weights or train in a separate copy.

## Measurement-loss comparison

`train_v4_measurement_ablation.py` runs the corresponding zero-measurement-loss branch. `evaluate_v4_measurement_ablation.py` generates the independent synthetic evaluation using the full and ablated checkpoints. It is a synthetic validation, not a new real experimental recording.

## EBIV and EBOS

`downstream_ebiv_context_v4.py` and `downstream_ebos_context_v4.py` preserve the archived measurement protocols. Their `--help` commands describe inputs. They require separately obtained solver repositories, native dependencies, recordings, configuration files and source-derived reference arrays. The historical `AMS2026 (2)/AMS_template_2022` directory structure is retained by some adapters.

The EBIV adapter still imports an external solver and the historical tracking helper. Those dependencies are not bundled here. Downstream wrappers are staged source, not a validated stand-alone installation. Do not describe this candidate as full one-command paper reproduction.

Original experiments include low-light degradation and a high-noise EBIV boundary. Hot-plate observations affected method development. Preserve these qualifications when reporting results.
