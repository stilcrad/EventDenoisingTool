"""Reproducible preparation, training and held-out event evaluation.

Run with the existing ebos Python environment. All outputs are below this folder.
Pilot and expanded runs are separate. Never fit on test or EBIV recordings.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from features import FEATURE_NAMES, GENERIC_COLUMNS, extract

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOS = ROOT / 'AMS2026 (2)' / 'AMS_template_2022' / '_codex_event_based_bos_extract' / 'event_based_bos-main'
DATA = Path(os.environ.get('VGS_DATA_ROOT', str(ROOT / 'data' / 'raw' / 'CCS')))
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def hash_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def search(ds, value):
    lo, hi = 0, len(ds)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(ds[mid]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def load_window(scene, fraction, cfg):
    path = DATA / scene / 'prophesee_0' / 'events.hdf5'
    with h5py.File(path, 'r') as f:
        g = f['raw_events']
        first, last = int(g['t'][0]), int(g['t'][-1])
        start = int(first + fraction * (last - first))
        stop = start + cfg['window_us']
        if stop > last:
            raise ValueError('Window outside source recording')
        i0, i1 = search(g['t'], start - cfg['warmup_us']), search(g['t'], stop)
        a = np.column_stack([g[k][i0:i1] for k in ('t', 'x', 'y', 'p')]).astype(np.int64)
    a[:, 3] = a[:, 3] > 0
    info = dict(source=str(path), source_bytes=path.stat().st_size,
                source_mtime_ns=path.stat().st_mtime_ns, source_count=len(g) if False else i1-i0,
                source_indices=[i0, i1], window_us=[start, stop],
                warmup_start_us=start-cfg['warmup_us'], source_slice_sha256=hash_array(a),
                x_range=[int(a[:, 1].min()), int(a[:, 1].max())],
                y_range=[int(a[:, 2].min()), int(a[:, 2].max())])
    return a, start, stop, info


def inject(source, rate, kind, seed, H=720, W=1280, hot_pixels=80):
    rng = np.random.default_rng(seed)
    n = int(round(len(source) * rate))
    noise = np.empty((n, 4), np.int64)
    noise[:, 0] = rng.integers(int(source[0, 0]), int(source[-1, 0])+1, n)
    if kind == 'uniform':
        noise[:, 1] = rng.integers(0, W, n)
        noise[:, 2] = rng.integers(0, H, n)
        noise[:, 3] = rng.integers(0, 2, n)
    elif kind == 'hot_pixel':
        # Locations are unique; each hot pixel keeps one polarity. This is a new
        # controlled protocol, not a claim of reproducing the old paper streams.
        pixels = rng.choice(H*W, min(hot_pixels, H*W), replace=False)
        pol = rng.integers(0, 2, len(pixels))
        ids = rng.integers(0, len(pixels), n)
        noise[:, 1], noise[:, 2], noise[:, 3] = pixels[ids] % W, pixels[ids] // W, pol[ids]
    else:
        raise ValueError(kind)
    ev = np.concatenate([source, noise])
    y = np.concatenate([np.ones(len(source), np.uint8), np.zeros(n, np.uint8)])
    order = np.argsort(ev[:, 0], kind='stable')
    return ev[order], y[order]


def prepare(out, pilot):
    cfg = copy.deepcopy(PROTOCOL)
    if pilot:
        cfg['train_window_fractions'] = [0.5]
        cfg['noise_rates'] = [0.2]
        cfg['noise_seeds'] = [7301]
    cfg['pilot'] = pilot
    groups = [set(x) for x in cfg['splits'].values()]
    assert all(not a.intersection(b) for i, a in enumerate(groups) for b in groups[i+1:])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == cfg, 'Existing config differs'
    dump(out / 'config.json', cfg)
    dump(out / 'environment.json', dict(python=sys.version, numpy=np.__version__,
         torch=torch.__version__, sklearn=sklearn.__version__, platform=platform.platform(),
         cuda=torch.cuda.is_available(), device='cpu', torch_threads=4))
    manifest = []
    extract(np.array([[1,0,0,1],[2,0,0,1]], np.int64), 2, 2)
    for split, scenes in cfg['splits'].items():
        fractions = cfg['train_window_fractions'] if split == 'train' else cfg['evaluation_window_fractions']
        for scene_id, scene in enumerate(scenes):
            for fraction in fractions:
                source, start, stop, info = load_window(scene, fraction, cfg)
                for kind in cfg['noise_types']:
                    for rate in cfg['noise_rates']:
                        # Repeated test corruption; training has multiple scenes,
                        # windows and noise rates, so one seed per case is enough.
                        seeds = cfg['noise_seeds'] if split == 'test' else cfg['noise_seeds'][:1]
                        for seed in seeds:
                            effective_seed = seed + scene_id * 100 + int(fraction*10000)
                            key = f'{split}_{scene}_{fraction:g}_{kind}_{rate:g}_{seed}'
                            path = out / 'cache' / f'{key}.npz'
                            meta_path = path.with_suffix('.json')
                            if path.exists() and meta_path.exists():
                                manifest.append(json.loads(meta_path.read_text()))
                                continue
                            t0 = time.perf_counter()
                            ev, y = inject(source, rate, kind, effective_seed,
                                           cfg['height'], cfg['width'], cfg['hot_pixels'])
                            features, scores = extract(ev, cfg['height'], cfg['width'])
                            feature_time = time.perf_counter()-t0
                            idx = np.flatnonzero((ev[:, 0] >= start) & (ev[:, 0] < stop))
                            n_active = len(idx)
                            if split != 'test':
                                limit = cfg['sample_per_train_case' if split == 'train' else 'sample_per_validation_case']
                                if len(idx) > limit:
                                    idx = np.sort(np.random.default_rng(effective_seed+90000).choice(idx, limit, replace=False))
                            path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(path, features=features[idx], scores=scores[idx], labels=y[idx])
                            meta = dict(key=key, split=split, scene=scene, fraction=fraction,
                                        noise_type=kind, rate=rate, seed=effective_seed, cache=str(path),
                                        n_input=len(ev), n_active=n_active, n_cached=len(idx),
                                        source_fraction=float(y[idx].mean()), preparation_s=feature_time,
                                        noisy_sha256=hash_array(ev), **info)
                            dump(meta_path, meta)
                            manifest.append(meta)
                            print(f'prepared {key}: {len(ev):,} events -> {len(idx):,} samples ({feature_time:.1f}s)', flush=True)
    dump(out / 'manifest.json', manifest)


def network(kind, n):
    if kind == 'linear':
        return nn.Linear(n, 1)
    return nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))


def predict(model, f, mean, scale, columns):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(f), 65536):
            x = np.clip((f[start:start+65536, columns]-mean)/scale, -20, 20).astype(np.float32)
            values.append(torch.sigmoid(model(torch.from_numpy(x))).flatten().numpy())
    return np.concatenate(values) if values else np.empty(0, np.float32)


def load_models(out):
    models = {}
    for file in sorted((out/'models').glob('*.pt')):
        ckpt = torch.load(file, map_location='cpu', weights_only=False)
        model = network(ckpt['kind'], len(ckpt['columns']))
        model.load_state_dict(ckpt['state_dict'])
        models[file.stem] = (model, ckpt)
    return models


def train(out):
    cfg = json.loads((out/'config.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    def gather(split):
        data = [np.load(m['cache']) for m in manifest if m['split'] == split]
        return np.concatenate([d['features'] for d in data]), np.concatenate([d['labels'] for d in data])
    xf, yf = gather('train')
    xv, yv = gather('validation')
    val_cases = [np.load(m['cache']) for m in manifest if m['split']=='validation']
    (out/'models').mkdir(exist_ok=True)
    torch.set_num_threads(4)
    for kind in ['linear', 'mlp_generic', 'mlp_physics']:
        columns = GENERIC_COLUMNS if kind == 'mlp_generic' else list(range(len(FEATURE_NAMES)))
        mean = xf[:, columns].mean(axis=0)
        scale = np.maximum(xf[:, columns].std(axis=0), 1e-6)
        xt = torch.from_numpy(np.clip((xf[:, columns]-mean)/scale, -20, 20).astype(np.float32))
        yt = torch.from_numpy(yf.astype(np.float32))[:,None]
        for seed in cfg['training_seeds']:
            target = out/'models'/f'{kind}_{seed}.pt'
            if target.exists():
                print(f'reuse {target.name}', flush=True)
                continue
            torch.manual_seed(seed)
            model = network(kind, len(columns))
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
            best, best_state, best_epoch = -1, None, 0
            history = []
            t0 = time.perf_counter()
            for epoch in range(cfg['epochs']):
                model.train()
                order = torch.randperm(len(xt))
                total = 0.0
                for offset in range(0, len(xt), cfg['batch_size']):
                    ix = order[offset:offset+cfg['batch_size']]
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy_with_logits(model(xt[ix]), yt[ix])
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())*len(ix)
                pv = predict(model, xv, mean, scale, columns)
                ap = float(average_precision_score(yv, pv))
                history.append(dict(epoch=epoch+1, loss=total/len(xt), validation_ap=ap))
                if ap > best:
                    best, best_state, best_epoch = ap, copy.deepcopy(model.state_dict()), epoch+1
            model.load_state_dict(best_state)
            # Most conservative case-specific threshold; never use test labels.
            thresholds = [float(np.quantile(predict(model, d['features'], mean, scale, columns)[d['labels']==1],
                              1-cfg['target_source_recall'], method='lower')) for d in val_cases]
            checkpoint = dict(kind=kind, seed=seed, columns=columns, feature_names=[FEATURE_NAMES[i] for i in columns],
                              mean=mean, scale=scale, threshold=min(thresholds), state_dict=best_state,
                              validation_ap=best, best_epoch=best_epoch, training_samples=len(xf),
                              parameters=sum(p.numel() for p in model.parameters()), training_s=time.perf_counter()-t0)
            torch.save(checkpoint, target)
            dump(target.with_suffix('.json'), {k:v for k,v in checkpoint.items() if k not in ['mean','scale','state_dict']})
            dump(target.with_suffix('.history.json'), history)
            print(f'trained {target.stem}: val AP={best:.5f}, threshold={min(thresholds):.5f}, {checkpoint["training_s"]:.1f}s', flush=True)
    vgs_threshold = min(float(np.quantile(d['scores'][d['labels']==1], 1-cfg['target_source_recall'], method='lower')) for d in val_cases)
    dump(out/'vgs_calibration.json', dict(threshold=vgs_threshold,
         note='Zeros retained if calibration threshold is zero; historical VGS positive-score gate reported separately'))


def metrics(y, p, keep):
    signal, noise = y == 1, y == 0
    tp, fp = int((keep & signal).sum()), int((keep & noise).sum())
    return dict(source_recall=tp/max(int(signal.sum()),1), noise_rejection=1-fp/max(int(noise.sum()),1),
                retention=float(keep.mean()), precision=tp/max(tp+fp,1),
                ap=float(average_precision_score(y,p)), auc=float(roc_auc_score(y,p)))


def evaluate(out):
    torch.set_num_threads(4)
    manifest = json.loads((out/'manifest.json').read_text())
    models = load_models(out)
    th = json.loads((out/'vgs_calibration.json').read_text())['threshold']
    rows = []
    for m in manifest:
        if m['split'] != 'test':
            continue
        d = np.load(m['cache'])
        f,y,s = d['features'],d['labels'],d['scores']
        base = {k:m[k] for k in ['scene','fraction','noise_type','rate','seed','n_cached']}
        for name, keep in [('raw', np.ones(len(y),bool)), ('vgs_positive',s>0), ('vgs_calibrated',s>=th)]:
            rows.append(dict(**base, method=name, inference_s=0, **metrics(y,s if name!='raw' else np.ones(len(y)),keep)))
        for name,(model,c) in models.items():
            t0 = time.perf_counter()
            p = predict(model,f,c['mean'],c['scale'],c['columns'])
            elapsed = time.perf_counter()-t0
            rows.append(dict(**base,method=name,inference_s=elapsed,**metrics(y,p,p>=c['threshold'])))
        print(f'evaluated {m["key"]}', flush=True)
    dump(out/'event_metrics.json', rows)
    with (out/'event_metrics.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','train','evaluate','all'])
    p.add_argument('--run', default='pilot_v1')
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    if Path(args.run).name!=args.run:
        raise ValueError('Run must be a directory name')
    out=HERE/'runs'/args.run
    for action in ['prepare','train','evaluate']:
        if args.action not in [action,'all']:
            continue
        if action=='prepare': prepare(out,args.pilot)
        elif action=='train': train(out)
        else: evaluate(out)


if __name__=='__main__':
    main()
