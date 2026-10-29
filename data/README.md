# Data preparation

No real experimental dataset has been redistributed in this directory. No public data DOI or CSTR is available yet.

`synthetic_example/events.npy` contains 2,000 deterministic artificial events with a 24 x 32 sensor and seed 20261029. It only tests the file interface. It is not the synthetic measurement-loss training dataset or an independent validation dataset.

Training expects the authorized EBOS recordings under `raw/CCS/<scene>/prophesee_0/events.hdf5`. The `raw_events` group contains arrays `t`, `x`, `y` and `p`. Alternatively set `VGS_DATA_ROOT` to the directory containing the scene folders.

The v2 protocol records the scenes, time-window fractions, noise seeds and splits used by the v4 trainer. These are provenance records, not proof that the entire development process was blind to all nominal test scenes.

Before public release, record the true original sources and versions, obtain redistribution permission where necessary, and identify every data file needed for the reported findings. Third-party datasets should be obtained from their authorized sources when redistribution is not permitted. An unfilled raw directory does not constitute an openly available dataset.
