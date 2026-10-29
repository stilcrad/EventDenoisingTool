"""Causal descriptors and a score-faithful, accelerated full VGS baseline.

Input columns: timestamp (us), x, y, polarity. No label or future event is read.
The historical vgs_core.py remains the reference and is not modified.
"""
import math
import numpy as np
from numba import njit

FEATURE_NAMES = [
    'log_dt_us', 'log_fixed_support', 'fixed_neighbor_fraction', 'polarity',
    'initialized', 'log_seen', 'ig_reliability', 'early_gate',
    'log_adaptive_support', 'log_weighted_support', 'stability_ratio',
    'log_dt_over_mu', 'log_interval_cv', 'log_mu_us',
]
GENERIC_COLUMNS = list(range(6))


@njit(cache=True)
def _extract(events, H, W):
    n = len(events)
    ts = np.full((H, W, 2), -1, np.int64)
    last = np.full((H, W, 2), -1e15)
    mu = np.full((H, W, 2), 5000.0)
    var = np.full((H, W, 2), 25e6)
    lam = np.ones((H, W, 2))
    llmean = np.zeros((H, W, 2))
    llvar = np.ones((H, W, 2))
    seen = np.zeros((H, W, 2), np.int32)
    f = np.zeros((n, 14), np.float32)
    scores = np.zeros(n, np.float32)
    inv2s = 1.0 / (2 * 1.8 * 1.8 + 1e-12)
    for i in range(n):
        t, x, y = events[i, 0], events[i, 1], events[i, 2]
        p = 1 if events[i, 3] > 0 else 0
        initialized = last[y, x, p] > -1e14
        dt = float(t - last[y, x, p]) if initialized else 0.0
        m = mu[y, x, p]
        v = var[y, x, p]
        tau = int(min(max(2.0 * m, 500.0), 10000.0))
        # ema_dt and mu_dt in the historical full implementation have identical
        # initialization and updates, so their ratio is exactly 1 until clipping.
        passed = dt >= min(max(m / max(m, 1.0), 0.2), 0.5) * m
        ri, gi, nll = 1.0, (1.0 if passed else 0.3), 0.0
        if initialized and passed:
            llambda = m ** 3 / max(v, 1.0)
            d = max(dt, 1e-12)
            mm = max(m, 1e-12)
            la = max(llambda, 1e-12)
            nll = -(0.5 * (math.log(la) - math.log(2 * math.pi) - 3 * math.log(d))
                    - la * (d - mm) ** 2 / (2 * mm * mm * d))
            z = (nll - llmean[y, x, p]) / math.sqrt(max(llvar[y, x, p], 1e-6))
            ri = 1.0 / (1.0 + math.exp(min(max((z - 4.0) * 2, -50.0), 50.0)))
        s0, sw, sf, nf = 0.0, 0.0, 0.0, 0.0
        for dy in range(-2, 3):
            yy = y + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-2, 3):
                xx = x + dx
                if xx < 0 or xx >= W:
                    continue
                tn = ts[yy, xx, p]
                if tn <= 0:
                    continue
                dtn = t - tn
                if dtn <= 0:
                    continue
                ws = math.exp(-float(dx * dx + dy * dy) * inv2s)
                if dtn <= 5000:
                    sf += ws * math.exp(-float(dtn * dtn) / (2 * 5000 ** 2 + 1e-12))
                    nf += 1
                if dtn <= tau:
                    w = ws * math.exp(-float(dtn * dtn) / (2 * tau * tau + 1e-12))
                    s0 += w
                    sw += w * (1.0 - math.exp(-max(lam[yy, xx, 0], lam[yy, xx, 1]) / 1000.0))
        f[i, 0] = math.log1p(dt)
        f[i, 1] = math.log1p(sf)
        f[i, 2] = nf / 25
        f[i, 3] = p
        f[i, 4] = initialized
        f[i, 5] = math.log1p(seen[y, x, p])
        f[i, 6] = ri if initialized else 0.0
        f[i, 7] = gi if initialized else 0.0
        f[i, 8] = math.log1p(s0)
        f[i, 9] = math.log1p(sw)
        f[i, 10] = sw / (s0 + 1e-12)
        f[i, 11] = math.log1p(dt / max(m, 1.0))
        f[i, 12] = math.log1p(math.sqrt(v) / max(m, 1.0))
        f[i, 13] = math.log1p(m)
        if initialized:
            scores[i] = sw * ri * gi
            mnew = 0.98 * m + 0.02 * dt
            mu[y, x, p] = mnew
            var[y, x, p] = max(0.98 * v + 0.02 * (dt - mnew) ** 2, 1.0)
            lam[y, x, p] = mnew ** 3 / var[y, x, p]
            llmean[y, x, p] = 0.98 * llmean[y, x, p] + 0.02 * nll
            llvar[y, x, p] = max(0.98 * llvar[y, x, p] + 0.02 * (nll - llmean[y, x, p]) ** 2, 1e-6)
        last[y, x, p] = t
        seen[y, x, p] += 1
        ts[y, x, p] = t
    return f, scores


def extract(events, H=720, W=1280):
    events = np.asarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError('Expected N x 4 (t,x,y,p)')
    if len(events):
        if np.any(np.diff(events[:, 0]) < 0):
            raise ValueError('Events must be time sorted')
        if (events[:, 1:3] < 0).any() or (events[:, 1] >= W).any() or (events[:, 2] >= H).any():
            raise ValueError('Coordinates outside sensor dimensions')
        if not np.isin(events[:, 3], [0, 1]).all():
            raise ValueError('Polarity must be normalized to 0/1')
    return _extract(events, H, W)
