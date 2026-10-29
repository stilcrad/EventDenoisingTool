"""Integrity checks against historical VGS plus causality and data alignment."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from vgs_core import vgs_scores
from features import extract


def main():
    rng = np.random.default_rng(81)
    n = 6000
    ev = np.column_stack([np.cumsum(rng.integers(0, 70, n)), rng.integers(0, 12, n),
                          rng.integers(0, 8, n), rng.integers(0, 2, n)])
    f, s = extract(ev, 8, 12)
    ref = vgs_scores(ev, H=8, W=12)
    np.testing.assert_allclose(s, ref, rtol=2e-6, atol=1e-7)
    pf, ps = extract(ev[:3123], 8, 12)
    np.testing.assert_array_equal(f[:3123], pf)
    np.testing.assert_array_equal(s[:3123], ps)
    assert np.isfinite(f).all() and (s >= 0).all()
    zf, zs = extract(np.empty((0, 4), np.int64), 8, 12)
    assert zf.shape == (0, 14) and zs.shape == (0,)
    try:
        extract(np.array([[0, 12, 0, 1]]), 8, 12)
        raise AssertionError('Invalid coordinate accepted')
    except ValueError:
        pass
    print('PASS: VGS parity, prefix causality, finite features, empty input, coordinate bounds')


if __name__ == '__main__':
    main()
