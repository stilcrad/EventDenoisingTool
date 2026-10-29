"""Event-centred time-surface patches with a declared fixed 5 ms lookahead.

This is a delayed, noncausal alternative to ``features_v2``, not an online
zero-delay feature extractor. It returns one row per original input event and
never changes the event's timestamp, coordinates, polarity, or order.

The 100 columns are 5x5 patches ordered as past/same, past/opposite,
future/same, future/opposite. Each patch is row-major (dy then dx), including
the centre. Values are log1p(lag_us)/log1p(5000), clipped to [0,1]; absent,
out-of-sensor, or more-than-5-ms history/context is 1. Equal-time events are
excluded even at other pixels: past/future timestamps must be strictly
smaller/larger. Only timestamps at most 5000 us away can affect a feature.

The final 5 ms of a finite recording is right-censored. Use
``complete_context_mask(events[:, 0])`` to restrict evaluation to rows whose
future observation window is complete. The feature extractor intentionally
does not drop these rows, which preserves alignment with the raw event stream.
"""
import math
from numbers import Integral

import numpy as np
from numba import njit


CONTEXT_US = 5000
LOOKAHEAD_US = CONTEXT_US
FEATURE_NAMES_CONTEXT = [
    f"{direction}_{polarity}_dy{dy:+d}_dx{dx:+d}_log_lag"
    for direction in ("past", "future")
    for polarity in ("same", "opposite")
    for dy in range(-2, 3)
    for dx in range(-2, 3)
]
PAST_CONTEXT_COLUMNS = list(range(50))
FUTURE_CONTEXT_COLUMNS = list(range(50, 100))
_MIN_TIME = np.iinfo(np.int64).min
_MAX_TIME = np.iinfo(np.int64).max


def _validated_events(events, H, W):
    if (not isinstance(H, Integral) or not isinstance(W, Integral)
            or isinstance(H, bool) or isinstance(W, bool) or H <= 0 or W <= 0):
        raise ValueError("Sensor dimensions must be positive integers")
    events = np.asarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Expected N x 4 (t,x,y,p)")
    if len(events):
        if np.any(events[1:, 0] < events[:-1, 0]):
            raise ValueError("Events must be time sorted")
        if ((events[:, 1:3] < 0).any() or (events[:, 1] >= W).any()
                or (events[:, 2] >= H).any()):
            raise ValueError("Coordinates outside sensor dimensions")
        if not np.isin(events[:, 3], [0, 1]).all():
            raise ValueError("Polarity must be normalized to 0/1")
        if np.isin(events[:, 0], [_MIN_TIME, _MAX_TIME]).any():
            raise ValueError("Timestamp collides with reserved context sentinel")
    return events


@njit(cache=True)
def _extract_context(events, H, W):
    f = np.ones((len(events), 100), np.float32)
    scale = math.log1p(CONTEXT_US)
    last = np.full((H, W, 2), _MIN_TIME, np.int64)
    previous = np.full((H, W, 2), _MIN_TIME, np.int64)
    for i in range(len(events)):
        t, x, y, p = events[i]
        for relation in range(2):
            polarity = p if relation == 0 else 1 - p
            for dy in range(-2, 3):
                yy = y + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in range(-2, 3):
                    xx = x + dx
                    if xx < 0 or xx >= W:
                        continue
                    tn = last[yy, xx, polarity]
                    if tn >= t:
                        tn = previous[yy, xx, polarity]
                    if tn == _MIN_TIME:
                        continue
                    lag = float(t) - float(tn)
                    if 0.0 < lag <= CONTEXT_US:
                        index = relation * 25 + (dy + 2) * 5 + dx + 2
                        f[i, index] = math.log1p(lag) / scale
        if last[y, x, p] != t:
            previous[y, x, p] = last[y, x, p]
            last[y, x, p] = t

    # Reuse the surfaces in reverse time; two distinct timestamps keep
    # simultaneous events from hiding valid strictly future context.
    last.fill(_MAX_TIME)
    previous.fill(_MAX_TIME)
    for i in range(len(events) - 1, -1, -1):
        t, x, y, p = events[i]
        for relation in range(2):
            polarity = p if relation == 0 else 1 - p
            for dy in range(-2, 3):
                yy = y + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in range(-2, 3):
                    xx = x + dx
                    if xx < 0 or xx >= W:
                        continue
                    tn = last[yy, xx, polarity]
                    if tn <= t:
                        tn = previous[yy, xx, polarity]
                    if tn == _MAX_TIME:
                        continue
                    lag = float(tn) - float(t)
                    if 0.0 < lag <= CONTEXT_US:
                        index = 50 + relation * 25 + (dy + 2) * 5 + dx + 2
                        f[i, index] = math.log1p(lag) / scale
        if last[y, x, p] != t:
            previous[y, x, p] = last[y, x, p]
            last[y, x, p] = t
    return f


def extract_context(events, H=720, W=1280):
    """Return an aligned float32 ``(N,100)`` array; requires 5 ms delay.

    Input uses integer microseconds and ``(t,x,y,p)`` with polarity 0/1.
    Missing context at recording boundaries is encoded as 1. Call
    ``complete_context_mask`` before evaluating the complete-window subset.
    """
    events = _validated_events(events, H, W)
    return _extract_context(events, int(H), int(W))


def complete_context_mask(timestamps):
    """True where at least 5000 us remains until the last observed timestamp.

    This addresses right censoring only. The earliest recording rows may still
    lack 5 ms of past history, as in a streaming cold start. Empty input yields
    an empty bool array. The last event time is the conservative observation
    endpoint; no unrecorded period is assumed.
    """
    t = np.asarray(timestamps, dtype=np.int64)
    if t.ndim != 1:
        raise ValueError("Expected one-dimensional timestamps")
    if len(t) == 0:
        return np.empty(0, dtype=bool)
    if np.any(t[1:] < t[:-1]):
        raise ValueError("Timestamps must be time sorted")
    # Python integer subtraction avoids int64 underflow near its minimum.
    end_of_complete_context = int(t[-1]) - CONTEXT_US
    if end_of_complete_context < _MIN_TIME:
        return np.zeros(len(t), dtype=bool)
    return t <= end_of_complete_context
