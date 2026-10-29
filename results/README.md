# Archived results

The supplied CSV files are unchanged copies from the learning-after archive. They are not newly computed results.

- `context_v4_augmented/validation_metrics.csv` contains event-level validation metrics.
- `v4_measurement_audit/ebos_paired_summary.csv` and `ebos_seed_differences.csv` summarize the paired measurement-loss ablation.
- `v4_measurement_audit/independent_synthetic_cases.csv` and `independent_synthetic_summary.csv` contain the independent synthetic check, including unfavorable results.

The EBIV/EBOS downstream CSVs and detailed paired rows still contain local filesystem paths. They have not yet been sanitized or copied into this release candidate. This is an explicit packaging gap, not a decision to omit negative scenes.

The complete source-derived measurement references and raw recordings are not included. AUC, displacement consistency and absolute physical accuracy are different claims. Do not relabel a source-derived reference as independent physical ground truth.
