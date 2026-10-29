#!/usr/bin/env python3
"""
Semi-synthetic EBIV denoising experiment on wallflow4_dense_3_100ms.evt.

The script adds uniformly distributed random events at fixed rates, applies a
conservative local spatio-temporal support filter in the EBIV-active time
interval, and then runs the same EBIV sum-of-correlation parameters for Clean,
Noisy, and Ours. Clean EBIV vectors are used as the reference for EPE-style
metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
EBIV_PY = HERE / "ebiv-main" / "python"
SAMPLE_DATA = HERE / "ebiv-main" / "sample_data"
DEFAULT_EVT = SAMPLE_DATA / "wallflow4_dense_3_100ms.evt"
DEFAULT_CFG = SAMPLE_DATA / "ebiv_params.cfg"
DEFAULT_OUT = HERE / "ebiv_noise_experiment"

sys.path.insert(0, str(EBIV_PY))
sys.path.insert(0, str(HERE))

from ebiv_params import EBIV_Config  # noqa: E402
from ebiv_utils import EBIDataSet, EBILoadEvents, FFT_CrossCorr2D, FindCorrPeak2D  # noqa: E402
from tracking_consistency_experiment import blockwise_keep_mask, vgs_scores  # noqa: E402


NOISE_RATES = (0.10, 0.20, 0.40)
LOW_PEAK_THRESHOLD = 0.10
OUTLIER_ABS_THRESHOLD = 0.50
OUTLIER_REL_THRESHOLD = 0.20
SUPPORT_RADIUS = 2
SUPPORT_TAU_US = 5000
SUPPORT_MIN_NEIGHBORS = 1


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_evt(path: Path) -> dict[str, np.ndarray | int | list[int]]:
    ev = EBILoadEvents(str(path))
    return {
        "t": ev["t"].astype(np.uint32, copy=False),
        "x": ev["x"].astype(np.uint16, copy=False),
        "y": ev["y"].astype(np.uint16, copy=False),
        "p": ev["p"].astype(np.uint8, copy=False),
        "time_stamp": int(ev["time_stamp"]),
        "image_size": [int(ev["image_size"][0]), int(ev["image_size"][1])],
    }


def save_evt(path: Path, ev: dict[str, np.ndarray | int | list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(ev["t"], dtype=np.uint32)
    x = np.asarray(ev["x"], dtype=np.uint32)
    y = np.asarray(ev["y"], dtype=np.uint32)
    p = np.asarray(ev["p"], dtype=np.uint32) & 1
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    x = x[order]
    y = y[order]
    p = p[order]

    event_count = int(t.size)
    word0 = ((x & 0xFFFF) | ((y & 0xFFFF) << 16)).astype(np.uint32)
    word1 = (((t.astype(np.uint64) << 1) | p.astype(np.uint64)) & 0xFFFFFFFF).astype(np.uint32)
    data = np.empty((event_count, 2), dtype=np.uint32)
    data[:, 0] = word0
    data[:, 1] = word1

    hdr_len = 64
    file_size = hdr_len + event_count * 8
    duration = int(t.max()) if event_count else 0
    h, w = [int(v) for v in ev["image_size"]]
    header = struct.pack(
        "4s3Q4L2Q",
        b"EVT3",
        file_size,
        event_count,
        int(ev["time_stamp"]),
        duration,
        hdr_len,
        w,
        h,
        0,
        0,
    )
    with path.open("wb") as f:
        f.write(header)
        data.tofile(f)


def event_dict_to_array(ev: dict[str, np.ndarray | int | list[int]], indices: np.ndarray | None = None) -> np.ndarray:
    if indices is None:
        return np.column_stack([ev["t"], ev["x"], ev["y"], ev["p"]]).astype(np.int64, copy=False)
    return np.column_stack([
        np.asarray(ev["t"])[indices],
        np.asarray(ev["x"])[indices],
        np.asarray(ev["y"])[indices],
        np.asarray(ev["p"])[indices],
    ]).astype(np.int64, copy=False)


def subset_event_dict(ev: dict[str, np.ndarray | int | list[int]], keep: np.ndarray) -> dict:
    return {
        "t": np.asarray(ev["t"])[keep].astype(np.uint32, copy=False),
        "x": np.asarray(ev["x"])[keep].astype(np.uint16, copy=False),
        "y": np.asarray(ev["y"])[keep].astype(np.uint16, copy=False),
        "p": np.asarray(ev["p"])[keep].astype(np.uint8, copy=False),
        "time_stamp": int(ev["time_stamp"]),
        "image_size": [int(ev["image_size"][0]), int(ev["image_size"][1])],
    }


def add_random_noise(ev: dict, rate: float, rng: np.random.Generator) -> dict:
    h, w = [int(v) for v in ev["image_size"]]
    n_clean = int(np.asarray(ev["t"]).size)
    n_noise = int(round(rate * n_clean))
    t_min = int(np.asarray(ev["t"]).min())
    t_max = int(np.asarray(ev["t"]).max())

    t_noise = rng.integers(t_min, t_max + 1, size=n_noise, dtype=np.uint32)
    x_noise = rng.integers(0, w, size=n_noise, dtype=np.uint16)
    y_noise = rng.integers(0, h, size=n_noise, dtype=np.uint16)
    p_noise = rng.integers(0, 2, size=n_noise, dtype=np.uint8)

    noisy = {
        "t": np.concatenate([np.asarray(ev["t"], dtype=np.uint32), t_noise]),
        "x": np.concatenate([np.asarray(ev["x"], dtype=np.uint16), x_noise]),
        "y": np.concatenate([np.asarray(ev["y"], dtype=np.uint16), y_noise]),
        "p": np.concatenate([np.asarray(ev["p"], dtype=np.uint8), p_noise]),
        "time_stamp": int(ev["time_stamp"]),
        "image_size": [h, w],
    }
    order = np.argsort(noisy["t"], kind="mergesort")
    for key in ("t", "x", "y", "p"):
        noisy[key] = noisy[key][order]
    return noisy


def active_time_interval(cfg: EBIV_Config) -> tuple[int, int]:
    first_t = int(cfg.t_start)
    return first_t - int(cfg.t_sep) // 2, first_t + int(cfg.t_sample) + int(cfg.t_sep) // 2


def denoise_vgs_active(
    noisy: dict,
    active_start: int,
    active_end: int,
    score_quantile: float,
    block_size: int,
) -> tuple[dict, dict]:
    t = np.asarray(noisy["t"])
    segment_idx = np.flatnonzero(t < active_end)
    segment = event_dict_to_array(noisy, segment_idx)
    scores = vgs_scores(segment)
    keep_segment = blockwise_keep_mask(
        segment,
        scores,
        block_size=block_size,
        quantile=score_quantile,
    )

    in_active_segment = (segment[:, 0] >= active_start) & (segment[:, 0] < active_end)
    keep_all = np.ones(t.size, dtype=bool)
    keep_all[segment_idx[in_active_segment]] = keep_segment[in_active_segment]

    denoised = subset_event_dict(noisy, keep_all)
    stats = {
        "scored_events": int(segment.shape[0]),
        "active_input_events": int(in_active_segment.sum()),
        "active_retained_events": int(keep_segment[in_active_segment].sum()),
        "active_retention_ratio": float(keep_segment[in_active_segment].mean()) if np.any(in_active_segment) else 0.0,
        "score_quantile": float(score_quantile),
        "block_size": int(block_size),
    }
    return denoised, stats


def temporal_support_mask(
    segment: np.ndarray,
    image_size: list[int],
    radius: int,
    tau_us: int,
    min_neighbors: int,
) -> np.ndarray:
    """Keep events having local temporal support in either time direction."""
    h, w = [int(v) for v in image_size]
    n = int(segment.shape[0])
    prev_support = np.zeros(n, dtype=np.uint8)
    next_support = np.zeros(n, dtype=np.uint8)
    old_time = -10**12
    future_time = 10**12

    last = np.full((h, w, 2), old_time, dtype=np.int64)
    for i in range(n):
        t = int(segment[i, 0])
        x = int(segment[i, 1])
        y = int(segment[i, 2])
        p = 1 if int(segment[i, 3]) > 0 else 0
        support = 0
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= h:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if xx < 0 or xx >= w:
                    continue
                if t - int(last[yy, xx, p]) <= tau_us:
                    support += 1
                    if support >= min_neighbors:
                        break
            if support >= min_neighbors:
                break
        prev_support[i] = min(support, 255)
        last[y, x, p] = t

    nxt = np.full((h, w, 2), future_time, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        t = int(segment[i, 0])
        x = int(segment[i, 1])
        y = int(segment[i, 2])
        p = 1 if int(segment[i, 3]) > 0 else 0
        support = 0
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= h:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if xx < 0 or xx >= w:
                    continue
                if int(nxt[yy, xx, p]) - t <= tau_us:
                    support += 1
                    if support >= min_neighbors:
                        break
            if support >= min_neighbors:
                break
        next_support[i] = min(support, 255)
        nxt[y, x, p] = t

    return (prev_support.astype(np.int16) + next_support.astype(np.int16)) >= min_neighbors


def denoise_support_active(
    noisy: dict,
    active_start: int,
    active_end: int,
    radius: int,
    tau_us: int,
    min_neighbors: int,
) -> tuple[dict, dict]:
    t = np.asarray(noisy["t"])
    segment_idx = np.flatnonzero(t < active_end)
    segment = event_dict_to_array(noisy, segment_idx)
    keep_segment = temporal_support_mask(
        segment,
        [int(v) for v in noisy["image_size"]],
        radius=radius,
        tau_us=tau_us,
        min_neighbors=min_neighbors,
    )

    in_active_segment = (segment[:, 0] >= active_start) & (segment[:, 0] < active_end)
    keep_all = np.ones(t.size, dtype=bool)
    keep_all[segment_idx[in_active_segment]] = keep_segment[in_active_segment]
    denoised = subset_event_dict(noisy, keep_all)
    stats = {
        "scored_events": int(segment.shape[0]),
        "active_input_events": int(in_active_segment.sum()),
        "active_retained_events": int(keep_segment[in_active_segment].sum()),
        "active_retention_ratio": float(keep_segment[in_active_segment].mean()) if np.any(in_active_segment) else 0.0,
        "support_radius": int(radius),
        "support_tau_us": int(tau_us),
        "support_min_neighbors": int(min_neighbors),
    }
    return denoised, stats


def make_cfg(cfg_path: Path) -> EBIV_Config:
    cfg = EBIV_Config(str(cfg_path))
    cfg.evalMethod = "corr_sum"
    cfg.sampleStepX = 80
    cfg.sampleStepY = 80
    return cfg


def run_ebiv(evt_path: Path, out_npz: Path, cfg_path: Path) -> np.ndarray:
    cfg = make_cfg(cfg_path)
    ev = EBIDataSet(str(evt_path))
    img_h, img_w = ev.imageSize()
    sz_sample = [cfg.sampleW, cfg.sampleH]
    vel_data = []
    nx = ny = nt = 0

    for t_0 in range(cfg.t_start, cfg.t_end, cfg.t_step):
        time_sample = [t_0, t_0 + cfg.t_sample]
        time_sample1 = [time_sample[0] - cfg.t_sep // 2, time_sample[1] - cfg.t_sep // 2]
        time_sample2 = [time_sample[0] + cfg.t_sep // 2, time_sample[1] + cfg.t_sep // 2]
        nt += 1
        ny = 0
        for iy in range(0, img_h - cfg.sampleH + 1, cfg.sampleStepY):
            nx = 0
            ny += 1
            for ix in range(0, img_w - cfg.sampleW + 1, cfg.sampleStepX):
                pos = [ix, iy]
                nx += 1
                s1 = ev.sampleEvents(pos, sz_sample, time_sample1, polarity=cfg.polarity)
                s2 = ev.sampleEvents(pos, sz_sample, time_sample2, polarity=cfg.polarity)
                ev_cnt1 = int(s1.evData["time"].size)
                ev_cnt2 = int(s2.evData["time"].size)
                if ev_cnt1 == 0 or ev_cnt2 == 0:
                    vx = vy = peak = 0.0
                    ev_cnt = 0
                else:
                    ev_cnt = (ev_cnt1 + ev_cnt2) // 2
                    vol1 = s1.asVolume(cfg.NT)
                    vol2 = s2.asVolume(cfg.NT)
                    corr = FFT_CrossCorr2D(vol1, vol2)
                    corr2d = corr.sum(axis=0)
                    corr_dx, corr_dy, peak = FindCorrPeak2D(corr2d)
                    vx = (corr_dx / cfg.t_sep) * 1000.0
                    vy = (corr_dy / cfg.t_sep) * 1000.0
                vel_data.append([s1.centroid[0], img_h - s1.centroid[1], vx, vy, peak, ev_cnt])
        if cfg.doSingleTimeStep:
            break

    if not vel_data:
        raise RuntimeError(f"No EBIV vectors generated for {evt_path}")
    arr = np.asarray(vel_data, dtype=np.float64).reshape(nt, ny, nx, 6)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, arr)
    return arr


def vector_fields(arr: np.ndarray) -> dict[str, np.ndarray]:
    data = arr[0]
    return {
        "x": data[:, :, 0],
        "y": data[:, :, 1],
        "vx": data[:, :, 2],
        "vy": data[:, :, 3],
        "peak": data[:, :, 4],
        "count": data[:, :, 5],
    }


def compute_metrics(name: str, arr: np.ndarray, clean_arr: np.ndarray) -> dict:
    f = vector_fields(arr)
    c = vector_fields(clean_arr)
    valid = (
        np.isfinite(f["vx"]) & np.isfinite(f["vy"]) &
        np.isfinite(c["vx"]) & np.isfinite(c["vy"]) &
        (c["count"] > 0) & (f["count"] > 0)
    )
    epe = np.sqrt((f["vx"] - c["vx"]) ** 2 + (f["vy"] - c["vy"]) ** 2)
    clean_mag = np.sqrt(c["vx"] ** 2 + c["vy"] ** 2)
    outlier_thr = np.maximum(OUTLIER_ABS_THRESHOLD, OUTLIER_REL_THRESHOLD * clean_mag)
    if np.any(valid):
        epe_valid = epe[valid]
        outlier_rate = float(np.mean(epe[valid] > outlier_thr[valid]))
        epe_mean = float(np.mean(epe_valid))
        epe_median = float(np.median(epe_valid))
        epe_p90 = float(np.quantile(epe_valid, 0.90))
    else:
        outlier_rate = epe_mean = epe_median = epe_p90 = float("nan")
    peak_valid = f["peak"][np.isfinite(f["peak"])]
    return {
        "stream": name,
        "valid_vectors": int(valid.sum()),
        "total_vectors": int(valid.size),
        "epe_mean_px_per_ms": epe_mean,
        "epe_median_px_per_ms": epe_median,
        "epe_p90_px_per_ms": epe_p90,
        "outlier_vector_rate": outlier_rate,
        "mean_peak_value": float(np.mean(peak_valid)) if peak_valid.size else float("nan"),
        "low_peak_ratio": float(np.mean(peak_valid < LOW_PEAK_THRESHOLD)) if peak_valid.size else float("nan"),
        "mean_event_count": float(np.mean(f["count"][f["count"] > 0])) if np.any(f["count"] > 0) else 0.0,
    }


def arrow(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], fill, width: int = 2) -> None:
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=fill, width=width)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    left = (-uy, ux)
    size = 7.0
    a = (p1[0] - size * ux + 0.55 * size * left[0], p1[1] - size * uy + 0.55 * size * left[1])
    b = (p1[0] - size * ux - 0.55 * size * left[0], p1[1] - size * uy - 0.55 * size * left[1])
    draw.polygon([p1, a, b], fill=fill)


def draw_velocity_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    arr: np.ndarray,
    metric: dict,
    vmag_max: float,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    panel_w = x1 - x0
    panel_h = y1 - y0
    f = vector_fields(arr)
    mag = np.sqrt(f["vx"] ** 2 + f["vy"] ** 2)
    norm = np.clip(mag / max(vmag_max, 1e-6), 0.0, 1.0)
    tile = cv2.resize((norm * 255).astype(np.uint8), (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(tile, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(color, mode="RGB")
    canvas.paste(img, (x0, y0))

    draw.rectangle(box, outline=(35, 35, 35), width=2)
    draw.rectangle((x0, y0, x1, y0 + 34), fill=(255, 255, 255))
    draw.text((x0 + 8, y0 + 7), title, fill=(20, 24, 30), font=font(18, True))
    if metric["stream"] != "Clean":
        txt = (
            f"EPE {metric['epe_mean_px_per_ms']:.2f}, "
            f"out {100.0 * metric['outlier_vector_rate']:.1f}%, "
            f"peak {metric['mean_peak_value']:.3f}"
        )
    else:
        txt = f"Clean ref., peak {metric['mean_peak_value']:.3f}"
    draw.text((x0 + 8, y1 - 26), txt, fill=(255, 255, 255), font=font(14))

    img_h = 720.0
    img_w = 1280.0
    q_scale = 34.0
    for yy in range(f["x"].shape[0]):
        for xx in range(f["x"].shape[1]):
            px = x0 + float(f["x"][yy, xx]) / img_w * panel_w
            py = y1 - float(f["y"][yy, xx]) / img_h * panel_h
            vx = float(f["vx"][yy, xx])
            vy = float(f["vy"][yy, xx])
            if not np.isfinite(vx + vy):
                continue
            p1 = (px + vx * q_scale, py - vy * q_scale)
            arrow(draw, (px, py), p1, fill=(255, 255, 255), width=2)


def draw_colorbar(canvas: Image.Image, box: tuple[int, int, int, int], vmag_max: float) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    h = y1 - y0
    grad = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    grad = np.repeat(grad, x1 - x0, axis=1)
    color = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(color, mode="RGB"), (x0, y0))
    draw.rectangle(box, outline=(40, 40, 40), width=1)
    draw.text((x0 - 8, y0 - 24), "|v| [px/ms]", fill=(35, 35, 35), font=font(14, True))
    for frac, label in [(0.0, f"{vmag_max:.1f}"), (0.5, f"{0.5 * vmag_max:.1f}"), (1.0, "0")]:
        y = y0 + frac * h
        draw.line((x1, y, x1 + 5, y), fill=(35, 35, 35), width=1)
        draw.text((x1 + 9, y - 8), label, fill=(35, 35, 35), font=font(13))


def make_velocity_figure(results: dict[str, np.ndarray], metrics: dict[str, dict], out_png: Path, out_pdf: Path) -> None:
    keys = [
        "Clean",
        "Noisy 10%",
        "Ours 10%",
        "Noisy 20%",
        "Ours 20%",
        "Noisy 40%",
        "Ours 40%",
    ]
    mags = []
    for key in keys:
        f = vector_fields(results[key])
        mags.append(np.sqrt(f["vx"] ** 2 + f["vy"] ** 2).ravel())
    vmag_max = float(np.quantile(np.concatenate(mags), 0.98))
    vmag_max = max(vmag_max, 1.0)

    canvas = Image.new("RGB", (2150, 1320), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 34), "EBIV velocity maps under semi-synthetic random event noise", fill=(20, 24, 30), font=font(32, True))
    draw.text(
        (70, 78),
        "Identical EBIV parameters are used for Clean, Noisy, and Ours. Metrics are computed against the Clean velocity map.",
        fill=(78, 78, 78),
        font=font(17),
    )

    boxes = [
        (70, 135, 570, 416),
        (650, 135, 1150, 416),
        (1230, 135, 1730, 416),
        (70, 490, 570, 771),
        (650, 490, 1150, 771),
        (1230, 490, 1730, 771),
        (70, 845, 570, 1126),
    ]
    for key, box in zip(keys, boxes):
        draw_velocity_panel(canvas, box, key, results[key], metrics[key], vmag_max)

    table_box = (650, 845, 1730, 1126)
    draw.rectangle(table_box, fill=(255, 255, 255), outline=(40, 40, 40), width=2)
    draw.text((table_box[0] + 18, table_box[1] + 16), "Quantitative summary", fill=(20, 24, 30), font=font(22, True))
    headers = ["Stream", "EPE", "Outlier", "Mean peak", "Low-peak"]
    xs = [table_box[0] + 24, table_box[0] + 310, table_box[0] + 480, table_box[0] + 670, table_box[0] + 870]
    y = table_box[1] + 62
    for x, htxt in zip(xs, headers):
        draw.text((x, y), htxt, fill=(70, 70, 70), font=font(15, True))
    y += 30
    for key in keys:
        m = metrics[key]
        vals = [
            key,
            "0.000" if key == "Clean" else f"{m['epe_mean_px_per_ms']:.3f}",
            "0.0%" if key == "Clean" else f"{100.0 * m['outlier_vector_rate']:.1f}%",
            f"{m['mean_peak_value']:.3f}",
            f"{100.0 * m['low_peak_ratio']:.1f}%",
        ]
        for x, val in zip(xs, vals):
            draw.text((x, y), val, fill=(30, 30, 30), font=font(15))
        y += 27

    draw_colorbar(canvas, (1860, 180, 1900, 760), vmag_max)
    draw.text(
        (70, 1190),
        f"Outlier: EPE > max({OUTLIER_ABS_THRESHOLD:g} px/ms, "
        f"{100 * OUTLIER_REL_THRESHOLD:g}% of Clean magnitude). "
        f"Low-peak: EBIV peak < {LOW_PEAK_THRESHOLD:g}.",
        fill=(78, 78, 78),
        font=font(16),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, "PNG")
    canvas.save(out_pdf, "PDF", resolution=300.0)


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "noise_rate",
        "stream",
        "event_count",
        "active_event_count",
        "active_retention_ratio",
        "valid_vectors",
        "total_vectors",
        "epe_mean_px_per_ms",
        "epe_median_px_per_ms",
        "epe_p90_px_per_ms",
        "outlier_vector_rate",
        "mean_peak_value",
        "low_peak_ratio",
        "mean_event_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_latex_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{EBIV velocity-estimation robustness under semi-synthetic random event noise. EPE is computed with respect to the Clean EBIV velocity field.}",
        r"\label{tab:ebiv_noise_metrics}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Noise & Stream & EPE & Outlier & Mean peak & Low-peak \\",
        r"level & & (px/ms) & rate & value & ratio \\",
        r"\hline",
    ]
    for row in selected:
        stream_label = str(row["stream"])
        if stream_label == "Clean":
            noise = "0\\%"
            stream = "Clean"
        else:
            noise = f"{int(round(float(row['noise_rate']) * 100))}\\%"
            stream = "Noisy" if stream_label.startswith("Noisy") else "Ours"
        lines.append(
            f"{noise} & {stream} & "
            f"{float(row['epe_mean_px_per_ms']):.3f} & "
            f"{100.0 * float(row['outlier_vector_rate']):.1f}\\% & "
            f"{float(row['mean_peak_value']):.3f} & "
            f"{100.0 * float(row['low_peak_ratio']):.1f}\\% \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def active_count(ev: dict, active_start: int, active_end: int) -> int:
    t = np.asarray(ev["t"])
    return int(((t >= active_start) & (t < active_end)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evt", type=Path, default=DEFAULT_EVT)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--denoise-mode", choices=["support", "vgs_quantile"], default="support")
    parser.add_argument("--score-quantile", type=float, default=0.40)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--support-radius", type=int, default=SUPPORT_RADIUS)
    parser.add_argument("--support-tau-us", type=int, default=SUPPORT_TAU_US)
    parser.add_argument("--support-min-neighbors", type=int, default=SUPPORT_MIN_NEIGHBORS)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    evt_dir = out_dir / "evt"
    vel_dir = out_dir / "velocity_npz"
    fig_dir = out_dir / "figs"

    cfg = make_cfg(args.cfg)
    active_start, active_end = active_time_interval(cfg)
    clean_ev = load_evt(args.evt)
    print(f"Loaded Clean EVT: {np.asarray(clean_ev['t']).size:,} events")
    print(f"EBIV active interval: {active_start}..{active_end} us")

    results: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    table_rows: list[dict] = []
    denoise_stats: dict[str, dict] = {}

    print("Running EBIV: Clean")
    clean_arr = run_ebiv(args.evt, vel_dir / "clean.npz", args.cfg)
    results["Clean"] = clean_arr
    metrics["Clean"] = compute_metrics("Clean", clean_arr, clean_arr)
    table_rows.append({
        "noise_rate": 0.0,
        "stream": "Clean",
        "event_count": int(np.asarray(clean_ev["t"]).size),
        "active_event_count": active_count(clean_ev, active_start, active_end),
        "active_retention_ratio": 1.0,
        **metrics["Clean"],
    })

    for rate in NOISE_RATES:
        label = f"{int(rate * 100)}%"
        noisy_key = f"Noisy {label}"
        ours_key = f"Ours {label}"
        noisy_evt = evt_dir / f"wallflow4_random_noise_{int(rate * 100)}pct.evt"
        ours_evt = evt_dir / f"wallflow4_random_noise_{int(rate * 100)}pct_vgs.evt"

        print(f"Generating {label} random event noise")
        noisy_ev = add_random_noise(clean_ev, rate, rng)
        save_evt(noisy_evt, noisy_ev)

        print(f"Applying {args.denoise_mode} denoising: {label}")
        if args.denoise_mode == "vgs_quantile":
            ours_ev, stats = denoise_vgs_active(
                noisy_ev,
                active_start,
                active_end,
                score_quantile=args.score_quantile,
                block_size=args.block_size,
            )
        else:
            ours_ev, stats = denoise_support_active(
                noisy_ev,
                active_start,
                active_end,
                radius=args.support_radius,
                tau_us=args.support_tau_us,
                min_neighbors=args.support_min_neighbors,
            )
        denoise_stats[label] = stats
        save_evt(ours_evt, ours_ev)

        print(f"Running EBIV: {noisy_key}")
        noisy_arr = run_ebiv(noisy_evt, vel_dir / f"noisy_{int(rate * 100)}pct.npz", args.cfg)
        print(f"Running EBIV: {ours_key}")
        ours_arr = run_ebiv(ours_evt, vel_dir / f"ours_{int(rate * 100)}pct.npz", args.cfg)

        results[noisy_key] = noisy_arr
        results[ours_key] = ours_arr
        metrics[noisy_key] = compute_metrics(noisy_key, noisy_arr, clean_arr)
        metrics[ours_key] = compute_metrics(ours_key, ours_arr, clean_arr)

        table_rows.append({
            "noise_rate": rate,
            "stream": "Noisy",
            "event_count": int(np.asarray(noisy_ev["t"]).size),
            "active_event_count": active_count(noisy_ev, active_start, active_end),
            "active_retention_ratio": 1.0,
            **metrics[noisy_key],
        })
        table_rows.append({
            "noise_rate": rate,
            "stream": "Ours",
            "event_count": int(np.asarray(ours_ev["t"]).size),
            "active_event_count": active_count(ours_ev, active_start, active_end),
            "active_retention_ratio": stats["active_retention_ratio"],
            **metrics[ours_key],
        })

    write_metrics_csv(out_dir / "ebiv_noise_metrics.csv", table_rows)
    write_latex_table(out_dir / "ebiv_noise_table.tex", table_rows)
    with (out_dir / "ebiv_noise_metrics.json").open("w") as f:
        json.dump(
            {
                "input_evt": str(args.evt),
                "config_file": str(args.cfg),
                "seed": args.seed,
                "noise_rates": list(NOISE_RATES),
                "active_interval_us": [active_start, active_end],
                "ebiv_method": "corr_sum",
                "sample_step_xy": [80, 80],
                "low_peak_threshold": LOW_PEAK_THRESHOLD,
                "outlier_definition": {
                    "absolute_px_per_ms": OUTLIER_ABS_THRESHOLD,
                    "relative_to_clean_magnitude": OUTLIER_REL_THRESHOLD,
                },
                "denoiser": {
                    "mode": args.denoise_mode,
                    "score_quantile": args.score_quantile,
                    "block_size": args.block_size,
                    "support_radius": args.support_radius,
                    "support_tau_us": args.support_tau_us,
                    "support_min_neighbors": args.support_min_neighbors,
                    "denoise_stats": denoise_stats,
                },
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    make_velocity_figure(
        results,
        metrics,
        fig_dir / "ebiv_velocity_maps.png",
        fig_dir / "ebiv_velocity_maps.pdf",
    )
    print(f"Metrics CSV: {out_dir / 'ebiv_noise_metrics.csv'}")
    print(f"Metrics JSON: {out_dir / 'ebiv_noise_metrics.json'}")
    print(f"LaTeX table: {out_dir / 'ebiv_noise_table.tex'}")
    print(f"Velocity figure: {fig_dir / 'ebiv_velocity_maps.png'}")


if __name__ == "__main__":
    main()
