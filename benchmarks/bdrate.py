#!/usr/bin/env python3
"""Bjontegaard delta-rate between two `examples/sliver64_rd` TSVs.

Usage:
    bdrate.py <baseline.tsv> <test.tsv> [psnr_column]

Both files must carry the same (name, w, h, speed, mode) cells at the same
quantizers. For each cell it fits a cubic to (PSNR, log10(bitrate)) over that
cell's quantizer sweep, integrates both fits over the overlapping PSNR range,
and reports the average rate difference in percent -- negative means the test
arm needs fewer bits at equal quality. Pure stdlib (no numpy on this host);
the 4x4 normal equations are solved with Gaussian elimination.
"""

import math
import sys
from collections import defaultdict


def polyfit3(xs, ys):
    """Least-squares cubic y = c0 + c1 x + c2 x^2 + c3 x^3."""
    n = 4
    a = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a[i][j] = sum(x ** (i + j) for x in xs)
        a[i][n] = sum(y * x**i for x, y in zip(xs, ys))
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise ValueError("singular fit")
        a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def integrate(c, lo, hi):
    """Definite integral of the cubic with coefficients c over [lo, hi]."""

    def anti(x):
        return c[0] * x + c[1] * x**2 / 2 + c[2] * x**3 / 3 + c[3] * x**4 / 4

    return anti(hi) - anti(lo)


def bd_rate(base, test):
    """base/test are lists of (rate, psnr). Returns percent rate change."""
    b = sorted(base, key=lambda p: p[1])
    t = sorted(test, key=lambda p: p[1])
    if len(b) < 4 or len(t) < 4:
        return None
    lo = max(b[0][1], t[0][1])
    hi = min(b[-1][1], t[-1][1])
    if hi - lo <= 0:
        return None
    cb = polyfit3([p[1] for p in b], [math.log10(p[0]) for p in b])
    ct = polyfit3([p[1] for p in t], [math.log10(p[0]) for p in t])
    diff = (integrate(ct, lo, hi) - integrate(cb, lo, hi)) / (hi - lo)
    return (10**diff - 1) * 100.0


def load(path, psnr_col):
    rows = defaultdict(list)
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(hdr)}
        for line in f:
            f_ = line.rstrip("\n").split("\t")
            if len(f_) != len(hdr):
                continue
            key = (f_[ix["name"]], f_[ix["speed"]], f_[ix["mode"]])
            # `psnr_avg` and `bpp` are derived columns; the committed TSVs drop
            # them, so recompute the 6:1:1 average when it is not present.
            if psnr_col == "psnr_avg" and "psnr_avg" not in ix:
                psnr = (
                    6.0 * float(f_[ix["psnr_y"]])
                    + float(f_[ix["psnr_u"]])
                    + float(f_[ix["psnr_v"]])
                ) / 8.0
            else:
                psnr = float(f_[ix[psnr_col]])
            rows[key].append(
                (
                    float(f_[ix["bytes"]]),
                    psnr,
                    int(f_[ix["px_64x16"]]) + int(f_[ix["px_16x64"]]),
                    int(f_[ix["ms"]]),
                    int(f_[ix["w"]]),
                    int(f_[ix["h"]]),
                )
            )
    return rows


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(2)
    psnr_col = sys.argv[3] if len(sys.argv) == 4 else "psnr_y"
    base = load(sys.argv[1], psnr_col)
    test = load(sys.argv[2], psnr_col)
    keys = sorted(set(base) & set(test))
    print(f"cell\tpx_sliver_base\tpx_sliver_test\tbd_rate_{psnr_col}_pct\tms_base\tms_test")
    vals, live_vals = [], []
    tb = tt = 0
    for k in keys:
        b, t = base[k], test[k]
        r = bd_rate([(x[0], x[1]) for x in b], [(x[0], x[1]) for x in t])
        pb = sum(x[2] for x in b)
        pt = sum(x[2] for x in t)
        mb = sum(x[3] for x in b)
        mt = sum(x[3] for x in t)
        tb += mb
        tt += mt
        name = "/".join(k)
        print(
            f"{name}\t{pb}\t{pt}\t"
            + ("n/a" if r is None else f"{r:+.3f}")
            + f"\t{mb}\t{mt}"
        )
        if r is not None:
            vals.append(r)
            if pb or pt:
                live_vals.append(r)
    if vals:
        print(f"\ncells: {len(vals)}  (sliver-live: {len(live_vals)})")
        print(f"mean BD-rate ({psnr_col}), all cells: {sum(vals)/len(vals):+.3f}%")
        if live_vals:
            print(
                f"mean BD-rate ({psnr_col}), sliver-live cells only: "
                f"{sum(live_vals)/len(live_vals):+.3f}%"
            )
        print(f"encode time base {tb/1000:.1f}s -> test {tt/1000:.1f}s "
              f"({100.0*(tt-tb)/tb:+.2f}%)")


if __name__ == "__main__":
    main()
