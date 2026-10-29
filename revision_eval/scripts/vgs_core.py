# -*- coding: utf-8 -*-
"""
vgs_core.py — shared implementation for revision experiments (checklist Steps 3-5).
Methods (NN-temporal / Spatial-MAJ / EBF / VGS) adapted verbatim from the manuscript
source tree (eval_jaer_full.py, tracking_consistency_experiment.py), with explicit
H,W support and VGS ablation switches.

Variant flags for VGS (checklist Section 5.1):
  use_ri   : False -> R_i = 1        (Variant 2: w/o R_i)
  use_gi   : False -> G_i = 1        (Variant 3: w/o G_i, no early-arrival penalty)
  use_wts  : False -> w_TS(j) = 1    (Variant 4: w/o timing-stability weighting)
  all three False                    -> Variant 1: ST support only (pure EBF-like)
  selection global_only=True         -> Variant 5: theta_b = theta_g
  all defaults                       -> Variant 6: Full VGS
"""
import csv
import math
from collections import defaultdict, deque

import numpy as np

# ───────────────────────── data loading ─────────────────────────

def load_rocket_csv(path, max_events=None):
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.reader(f)):
            if len(r) < 4:
                continue
            x, y, p, t = int(float(r[0])), int(float(r[1])), int(float(r[2])), int(float(r[3]))
            rows.append((t, x, y, 1 if p > 0 else 0))
            if max_events is not None and len(rows) >= max_events:
                break
    a = np.array(rows, dtype=np.int64)  # t, x, y, p
    return a[a[:, 0].argsort(kind="mergesort")]

# ───────────────────────── baselines ─────────────────────────

def denoise_nn_temporal(events, tau_us=5000, H=720, W=1280):
    last = np.full((H, W), -1e15)
    keep = np.zeros(len(events), dtype=bool)
    for i in range(len(events)):
        t, x, y, p = events[i]
        x, y = int(x), int(y)
        if t - last[y, x] > tau_us:
            keep[i] = True
            last[y, x] = t
    return events[keep]

def denoise_spatial_maj(events, radius=2, min_n=2, tau_us=5000, H=720, W=1280):
    N = len(events)
    keep = np.zeros(N, dtype=bool)
    cs = radius * 2 + 1
    gc, gw = (W + cs - 1) // cs, (H + cs - 1) // cs
    g = [[deque() for _ in range(gc)] for _ in range(gw)]
    for i in range(N):
        t, x, y, p = events[i]
        x, y = int(x), int(y)
        gx, gy = x // cs, y // cs
        for dgy in (-1, 0, 1):
            for dgx in (-1, 0, 1):
                cx, cy = gx + dgx, gy + dgy
                if 0 <= cx < gc and 0 <= cy < gw:
                    while g[cy][cx] and t - g[cy][cx][0][0] > tau_us:
                        g[cy][cx].popleft()
        cnt = 0
        for dgy in (-1, 0, 1):
            for dgx in (-1, 0, 1):
                cx, cy = gx + dgx, gy + dgy
                if 0 <= cx < gc and 0 <= cy < gw:
                    for bt, bx, by, bp in g[cy][cx]:
                        if abs(x - bx) <= radius and abs(y - by) <= radius:
                            cnt += 1
                        if cnt >= min_n:
                            break
                if cnt >= min_n:
                    break
        if cnt >= min_n:
            keep[i] = True
        g[gy][gx].append((t, x, y, p))
    return events[keep]

def ig_logpdf(dt, mu, lam, eps=1e-12):
    dt = max(float(dt), eps)
    mu = max(float(mu), eps)
    lam = max(float(lam), eps)
    return (0.5 * (math.log(lam) - math.log(2.0 * math.pi) - 3.0 * math.log(dt))
            - (lam * (dt - mu) ** 2) / (2.0 * mu * mu * dt))

def denoise_ebf(events, r=2, sigma_s=1.8, gamma_t=2.0, tau_max=10000, H=720, W=1280):
    """EBF spatio-temporal support; returns events whose support score > 0."""
    N = len(events)
    ts_pos = np.full((H, W), -1, dtype=np.int64)
    ts_neg = np.full((H, W), -1, dtype=np.int64)
    keep = np.zeros(N, dtype=bool)
    scores = np.zeros(N, dtype=np.float32)
    mu_dt = np.full((H, W, 2), 5000.0)
    last_t = np.full((H, W, 2), -1e15, dtype=np.float64)
    ema_mu = 0.02
    inv2s = 1.0 / (2 * sigma_s * sigma_s + 1e-12)
    for i in range(N):
        t = int(events[i, 0]); x = int(events[i, 1]); y = int(events[i, 2])
        p_idx = 1 if int(events[i, 3]) > 0 else 0
        lt = last_t[y, x, p_idx]
        if lt <= -1e14:
            last_t[y, x, p_idx] = t
        if p_idx:
            ts_pos[y, x] = t
        else:
            ts_neg[y, x] = t
        if lt <= -1e14:
            continue
        dt = t - lt
        mu = mu_dt[y, x, p_idx]
        tau_loc = int(np.clip(gamma_t * mu, 500, tau_max))
        mu_dt[y, x, p_idx] = (1 - ema_mu) * mu + ema_mu * dt
        last_t[y, x, p_idx] = t
        ts = ts_pos if p_idx else ts_neg
        S = 0.0
        for dy in range(-r, r + 1):
            yy = y + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-r, r + 1):
                xx = x + dx
                if xx < 0 or xx >= W:
                    continue
                tn = int(ts[yy, xx])
                if tn <= 0:
                    continue
                dtn = t - tn
                if dtn <= 0 or dtn > tau_loc:
                    continue
                ws = math.exp(-float(dx * dx + dy * dy) * inv2s)
                wt = math.exp(-float(dtn * dtn) / (2 * tau_loc * tau_loc + 1e-12))
                S += ws * wt
        scores[i] = float(S)
        keep[i] = S > 0
        if p_idx:
            ts_pos[y, x] = t
        else:
            ts_neg[y, x] = t
    return events[keep], scores

# ───────────────────────── VGS with ablation switches ─────────────────────────

def vgs_scores(
    events,
    radius=2, sigma_s=1.8, gamma_t=2.0, tau_max=10000, k_sigma=4.0,
    H=720, W=1280,
    use_ri=True, use_gi=True, use_wts=True,
):
    """VGS continuous scores. R_i=IG weight, G_i=early-arrival penalty, w_TS=timing-stability weight."""
    N = len(events)
    ts_pos = np.full((H, W), -1, dtype=np.int64)
    ts_neg = np.full((H, W), -1, dtype=np.int64)
    last_t = np.full((H, W, 2), -1e15, dtype=np.float64)
    mu_dt = np.full((H, W, 2), 5000.0)
    var_dt = np.full((H, W, 2), 25e6)
    ema_dt = np.full((H, W, 2), 5000.0)
    lam_dt = np.full((H, W, 2), 1.0)
    ll_mean = np.zeros((H, W, 2))
    ll_var = np.ones((H, W, 2))
    seen = np.zeros((H, W, 2), dtype=np.int32)
    ema_mu, ema_var = 0.02, 0.02
    inv2s = 1.0 / (2 * sigma_s * sigma_s + 1e-12)
    scores = np.zeros(N, dtype=np.float32)

    for i in range(N):
        t = int(events[i, 0]); x = int(events[i, 1]); y = int(events[i, 2])
        p_idx = 1 if int(events[i, 3]) > 0 else 0
        lt = last_t[y, x, p_idx]
        if lt <= -1e14:
            last_t[y, x, p_idx] = t
            seen[y, x, p_idx] = 1
            if p_idx:
                ts_pos[y, x] = t
            else:
                ts_neg[y, x] = t
            continue
        dt = t - lt
        mu_old = float(mu_dt[y, x, p_idx])

        # refractory / early-arrival gate (G_i)
        pass_refrac = True
        if use_gi:
            ratio = float(ema_dt[y, x, p_idx] / max(mu_old, 1.0))
            alpha_loc = float(np.clip(ratio, 0.2, 0.5))
            pass_refrac = dt >= alpha_loc * mu_old

        # IG reliability weight (R_i)
        nll = 0.0
        ig_weight = 1.0
        if use_ri and seen[y, x, p_idx] >= 1 and pass_refrac:
            var_loc = float(max(var_dt[y, x, p_idx], 1.0))
            lam_loc = mu_old ** 3 / var_loc
            nll = -ig_logpdf(dt, mu_old, lam_loc)
            mean_loc = float(ll_mean[y, x, p_idx])
            std_loc = math.sqrt(max(float(ll_var[y, x, p_idx]), 1e-6))
            z = (nll - mean_loc) / max(std_loc, 1e-6)
            z_clipped = float(np.clip((z - k_sigma) * 2.0, -50.0, 50.0))
            ig_weight = 1.0 / (1.0 + math.exp(z_clipped))

        # spatio-temporal support with timing-stability weight (w_TS)
        tau_loc = int(np.clip(gamma_t * mu_old, 500, tau_max))
        ts = ts_pos if p_idx else ts_neg
        S = 0.0
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if xx < 0 or xx >= W:
                    continue
                tn = int(ts[yy, xx])
                if tn <= 0:
                    continue
                dtn = t - tn
                if dtn <= 0 or dtn > tau_loc:
                    continue
                ws = math.exp(-float(dx * dx + dy * dy) * inv2s)
                wt = math.exp(-float(dtn * dtn) / (2 * tau_loc * tau_loc + 1e-12))
                if use_wts:
                    lam_n = max(float(lam_dt[yy, xx, 0]), float(lam_dt[yy, xx, 1]))
                    n_wig = 1.0 - math.exp(-lam_n / 1000.0)
                else:
                    n_wig = 1.0
                S += ws * wt * n_wig

        # final score: R_i * G_i * S
        gi = (1.0 if pass_refrac else 0.3) if use_gi else 1.0
        ri = ig_weight if use_ri else 1.0
        scores[i] = float(S * ri * gi)

        # state updates (always, as in the full implementation)
        m_new = (1.0 - ema_mu) * mu_old + ema_mu * dt
        mu_dt[y, x, p_idx] = m_new
        v_old = float(var_dt[y, x, p_idx])
        var_dt[y, x, p_idx] = max((1.0 - ema_var) * v_old + ema_var * (dt - m_new) ** 2, 1.0)
        lam_dt[y, x, p_idx] = m_new ** 3 / max(float(var_dt[y, x, p_idx]), 1.0)
        ema_dt[y, x, p_idx] = (1.0 - ema_mu) * float(ema_dt[y, x, p_idx]) + ema_mu * dt
        if use_ri:
            llm = float(ll_mean[y, x, p_idx])
            ll_mean[y, x, p_idx] = (1.0 - ema_mu) * llm + ema_mu * nll
            ll_var[y, x, p_idx] = max(
                (1.0 - ema_var) * float(ll_var[y, x, p_idx])
                + ema_var * (nll - float(ll_mean[y, x, p_idx])) ** 2, 1e-6)
        last_t[y, x, p_idx] = t
        seen[y, x, p_idx] += 1
        if p_idx:
            ts_pos[y, x] = t
        else:
            ts_neg[y, x] = t
    return scores

def blockwise_keep_mask(events, scores, block_size=64, quantile=0.70,
                        min_block_events=50, global_only=False, H=720, W=1280):
    nbx = (W + block_size - 1) // block_size
    block_ids = (events[:, 2] // block_size) * nbx + (events[:, 1] // block_size)
    positive = scores > 0
    global_thr = float(np.quantile(scores[positive], quantile)) if np.any(positive) else 0.0
    thresholds = defaultdict(lambda: global_thr)
    if not global_only:
        for bid in np.unique(block_ids):
            mask = (block_ids == bid) & positive
            if int(mask.sum()) >= min_block_events:
                thresholds[int(bid)] = float(np.quantile(scores[mask], quantile))
    keep = np.zeros(len(events), dtype=bool)
    for bid in np.unique(block_ids):
        keep[block_ids == bid] = scores[block_ids == bid] >= thresholds[int(bid)]
    keep &= positive
    return keep

def denoise_vgs(events, variant="full", H=720, W=1280, **kw):
    """variant in {full, st_only, no_ri, no_gi, no_wts, global_only}"""
    flags = {
        "full":        dict(use_ri=True,  use_gi=True,  use_wts=True,  global_only=False),
        "st_only":     dict(use_ri=False, use_gi=False, use_wts=False, global_only=False),
        "no_ri":       dict(use_ri=False, use_gi=True,  use_wts=True,  global_only=False),
        "no_gi":       dict(use_ri=True,  use_gi=False, use_wts=True,  global_only=False),
        "no_wts":      dict(use_ri=True,  use_gi=True,  use_wts=False, global_only=False),
        "global_only": dict(use_ri=True,  use_gi=True,  use_wts=True,  global_only=True),
    }
    if variant not in flags:
        raise ValueError(f"unknown variant {variant}")
    f = flags[variant]
    scores = vgs_scores(events, use_ri=f["use_ri"], use_gi=f["use_gi"],
                        use_wts=f["use_wts"], H=H, W=W, **kw)
    keep = blockwise_keep_mask(events, scores, global_only=f["global_only"], H=H, W=W)
    return events[keep], scores

# ───────────────────────── metrics ─────────────────────────

def table_metrics(events, raw_events, signal_roi, background_roi, H=720, W=1280):
    """event_count, retention, sparsity, ROI-based SBR. All on fixed 1280x720 grid."""
    n = len(events)
    act = np.zeros((H, W), dtype=bool)
    act[events[:, 2], events[:, 1]] = True
    sparsity = 1.0 - float(act.sum()) / (H * W)
    sx0, sy0, sx1, sy1 = signal_roi
    bx0, by0, bx1, by1 = background_roi
    in_sig = (events[:, 1] >= sx0) & (events[:, 1] < sx1) & (events[:, 2] >= sy0) & (events[:, 2] < sy1)
    in_bg = (events[:, 1] >= bx0) & (events[:, 1] < bx1) & (events[:, 2] >= by0) & (events[:, 2] < by1)
    n_sig, n_bg = int(in_sig.sum()), int(in_bg.sum())
    sbr = 10.0 * math.log10(n_sig / max(n_bg, 1)) if n_bg > 0 else float("inf")
    return {
        "event_count": n,
        "retention_ratio": n / max(len(raw_events), 1),
        "sparsity": sparsity,
        "n_signal_roi": n_sig,
        "n_background_roi": n_bg,
        "sbr_db": sbr,
    }
