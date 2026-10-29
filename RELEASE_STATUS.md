# Release status

Status: local staging, not authorized for public release.

## Included

- The v4 dependency chain, analytical VGS reference and original benchmark wrappers.
- Six archived checkpoints. Only the `training_protocol` path metadata was changed. Model tensors, normalization, thresholds and all remaining payload values were verified equal to the source checkpoints.
- Five archived numerical CSV files without machine-specific paths.
- A deterministic artificial event array for smoke testing.

## Required before publication

- Confirm authorship, institutional approval and redistribution rights for code, weights and derived data.
- Choose and add the actual code license. Decide whether model weights and data need separate licenses.
- Review dependency and third-party solver licenses. No license for this project can override another project's terms.
- Resolve dataset access, versions, recording identifiers, windows and lawful redistribution. Add actual repository identifiers when available.
- Sanitize and stage the omitted EBIV/EBOS CSV files while preserving all numerical entries. They remain in the local archive, not this repository.
- Test installation in a clean environment and reproduce the complete training and downstream evaluation workflows.
- Audit all manuscript result-to-file mappings, including historical non-learning tables.
- Review the final repository for sensitive information and credentials before creating any public remote.

The known broken historical `merge_results.py` files were excluded, not repaired. They are not dependencies of the staged core. No numerical experiment has been rerun or improved by packaging.

Only the `vgs-event-reliability` directory is a repository candidate. Its sibling `_private_review_DO_NOT_UPLOAD` contains local provenance and must remain private.
