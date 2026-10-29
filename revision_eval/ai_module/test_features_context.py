"""Hand-calculated and alignment checks for the fixed-5-ms-delay branch."""
import math

import numpy as np

from features_context import (CONTEXT_US, FEATURE_NAMES_CONTEXT,
                              complete_context_mask, extract_context)


def idx(direction, polarity, dy=0, dx=0):
    return FEATURE_NAMES_CONTEXT.index(
        f"{direction}_{polarity}_dy{dy:+d}_dx{dx:+d}_log_lag")


def scaled(lag):
    return math.log1p(lag) / math.log1p(CONTEXT_US)


def main():
    # The query event at t=10000 lies at (2,2), polarity 1. Repeated
    # timestamp/same-pixel neighbours exercise distinct-time state tracking.
    events = np.array([
        [4999, 0, 0, 1], [5000, 0, 1, 1], [9800, 2, 2, 1],
        [9850, 2, 2, 0], [9900, 1, 2, 1], [9990, 3, 2, 0],
        [10000, 2, 2, 1], [10000, 2, 2, 1], [10000, 1, 2, 1],
        [10010, 3, 2, 0], [10100, 1, 2, 1], [10150, 2, 2, 0],
        [10200, 2, 2, 1], [15000, 0, 1, 1], [15001, 0, 0, 1],
    ], np.int64)
    before = events.copy()
    f = extract_context(events, 5, 5)
    np.testing.assert_array_equal(events, before)
    assert f.shape == (len(events), 100) and f.dtype == np.float32
    assert len(FEATURE_NAMES_CONTEXT) == len(set(FEATURE_NAMES_CONTEXT)) == 100
    query = f[6]
    for direction in ("past", "future"):
        np.testing.assert_allclose(query[idx(direction, "same")], scaled(200))
        np.testing.assert_allclose(query[idx(direction, "opposite")], scaled(150))
        np.testing.assert_allclose(query[idx(direction, "same", dx=-1)], scaled(100))
        np.testing.assert_allclose(query[idx(direction, "opposite", dx=1)], scaled(10))
        assert query[idx(direction, "same", dy=-1, dx=-2)] == 1  # 5000 us
        assert query[idx(direction, "same", dy=-2, dx=-2)] == 1  # 5001 us
    np.testing.assert_array_equal(f[6], f[7])

    # Strictly simultaneous events, including a repeated centre event,
    # provide neither past nor future support.
    simultaneous = np.array([[100, 1, 1, 1], [100, 1, 1, 1],
                              [100, 0, 0, 0]], np.int64)
    assert (extract_context(simultaneous, 2, 2) == 1).all()
    edge = extract_context(np.array([[0, 0, 0, 0], [10, 1, 1, 0]]), 2, 2)
    assert edge[0, idx("future", "same", dy=-1, dx=-1)] == 1
    np.testing.assert_allclose(edge[0, idx("future", "same", dy=1, dx=1)], scaled(10))
    assert (extract_context(np.array([[100, 0, 0, 1]]), 1, 1) == 1).all()

    rng = np.random.default_rng(379)
    n = 1500
    random_events = np.column_stack((np.cumsum(rng.integers(0, 100, n)),
                                     rng.integers(0, 12, n), rng.integers(0, 8, n),
                                     rng.integers(0, 2, n)))
    rf = extract_context(random_events, 8, 12)
    # Past is causal for every prefix. Full past+future patches become final
    # after their declared 5 ms lookahead has been observed.
    for stop in (1, 401, 991):
        prefix = random_events[:stop]
        pf = extract_context(prefix, 8, 12)
        np.testing.assert_array_equal(pf[:, :50], rf[:stop, :50])
        complete = complete_context_mask(prefix[:, 0])
        np.testing.assert_array_equal(pf[complete], rf[:stop][complete])
    # Even events at the same spatial sites cannot affect rows more than
    # 5000 us earlier. Future 5-ms context is the only source of lookahead.
    distant = np.array([[events[-1, 0] + 6000, 2, 2, 1]], np.int64)
    extended = extract_context(np.concatenate((events, distant)), 5, 5)
    np.testing.assert_array_equal(extended[:len(events)], f)
    flipped = random_events.copy()
    flipped[:, 3] = 1 - flipped[:, 3]
    np.testing.assert_array_equal(extract_context(flipped, 8, 12), rf)
    assert np.isfinite(rf).all() and ((rf >= 0) & (rf <= 1)).all()
    np.testing.assert_array_equal(complete_context_mask([0, 1000, 5000, 6000]),
                                  [True, True, False, False])
    assert complete_context_mask([]).shape == (0,)
    assert extract_context(np.empty((0, 4), np.int64), 8, 12).shape == (0, 100)
    for invalid, h, w in [
        (np.array([[0, 12, 0, 1]]), 8, 12),
        (np.array([[2, 0, 0, 1], [1, 0, 0, 1]]), 8, 12),
        (np.array([[0, 0, 0, 2]]), 8, 12),
        (np.array([[0, 0, 0]]), 8, 12),
        (np.empty((0, 4)), 8, 0),
    ]:
        try:
            extract_context(invalid, h, w)
        except ValueError:
            continue
        raise AssertionError("Invalid event input accepted")
    for invalid in ([[0, 1]], [2, 1]):
        try:
            complete_context_mask(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid timestamps accepted")
    print("PASS: 100 context features; fixed 5 ms lookahead; strictly distinct "
          "times; centre exclusion of current event; hand calculations; "
          "past prefix causality; complete-window invariance; polarity "
          "symmetry; alignment; border/missing handling; right-censoring mask")


if __name__ == "__main__":
    main()
