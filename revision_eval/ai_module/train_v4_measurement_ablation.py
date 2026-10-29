"""Matched zero measurement-gradient ablation; retain Adam step and data."""
import train_context_v4_augmented as full

base = full.base
base.RUN = full.HERE / 'runs' / 'v4_no_measurement'
base.PROTOCOL_PATH = full.HERE / 'v4_measurement_ablation_protocol.json'
base.MEASUREMENT_WEIGHT = 0.0

if __name__ == '__main__':
    if (base.RUN / 'frozen_manifest.json').exists():
        raise RuntimeError('Already completed; refusing overwrite')
    base.main()
