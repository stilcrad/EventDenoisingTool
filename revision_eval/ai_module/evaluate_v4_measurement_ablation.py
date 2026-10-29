"""Independent IID synthetic evaluation, paired by realization and model seed."""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import train_context_v4_augmented as full
import train_context_v2 as v2

ROOT = full.HERE
OUT = ROOT / 'runs' / 'v4_measurement_audit'

def error(case, weights):
    centers = []
    for mask in (case['phase1'], case['phase2']):
        w = weights[mask]
        if w.sum() <= 0:
            return float('nan')
        centers.append((w[:, None] * case['coordinates'][mask]).sum(0) / w.sum())
    return float(np.linalg.norm(centers[1] - centers[0] - case['target']))

def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], []
    for seed in (6101, 6102, 6103):
        for name, directory in [('full', 'context_v4_augmented'), ('no_measurement', 'v4_no_measurement')]:
            path = ROOT / 'runs' / directory / 'models' / f'context_v3_{seed}.pt'
            payload, _, _, model, _, _, _, norm = v2.restore(path)
            cases = full.synthetic_measurement_cases(norm, count=128, seed=912031)
            threshold = payload['thresholds']['physics_context_mlp']
            provenance.append({'variant': name, 'seed': seed, 'checkpoint': str(path), 'sha256': v2.file_hash(path), 'threshold': threshold})
            for i, case in enumerate(cases):
                with torch.no_grad():
                    p = torch.sigmoid(model(torch.from_numpy(case['features']))).numpy().reshape(-1)
                keep = p >= threshold
                rows.append({'variant': name, 'seed': seed, 'case': i,
                    'raw_epe_px': error(case, np.ones_like(p)),
                    'soft_epe_px': error(case, p), 'hard_epe_px': error(case, keep.astype(np.float32)),
                    'source_recall': float(keep[case['labels'] > .5].mean()),
                    'noise_rejection': float((~keep[case['labels'] < .5]).mean())})
    data = pd.DataFrame(rows)
    data.to_csv(OUT / 'independent_synthetic_cases.csv', index=False)
    summary = data.groupby(['variant','seed'])[['raw_epe_px','soft_epe_px','hard_epe_px','source_recall','noise_rejection']].mean()
    summary.to_csv(OUT / 'independent_synthetic_summary.csv')
    paired = []
    for metric in ('soft_epe_px', 'hard_epe_px'):
        pivot = data.pivot(index=['seed','case'], columns='variant', values=metric)
        delta = (pivot['full']-pivot['no_measurement']).groupby('case').mean().to_numpy()
        rng = np.random.default_rng(913031)
        boot = np.mean(rng.choice(delta, size=(4000,len(delta)), replace=True),axis=1)
        paired.append({'metric':metric,'full_minus_ablation_px':float(delta.mean()),
            'case_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
            'note':'Average across training seeds per realization; bootstrap independent synthetic cases, not pixels'})
    (OUT / 'independent_synthetic_provenance.json').write_text(json.dumps({'seed':912031,'cases':128,'scope':'new IID synthetic cases only; no independent real recording claim','models':provenance,'paired':paired},indent=2),encoding='utf-8')
    print(summary.to_string())
    print(json.dumps(paired,indent=2))

if __name__ == '__main__':
    main()
