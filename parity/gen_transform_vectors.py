"""Parity vectors for the transforms that read or emit non-mixture predictives:
seasonal_scale, gaussianize, log_transform, the exact flag on power_transform
and yeo_johnson, stacked pushforwards, flesaker, and laplace's tail-spliced
predictive through each component-reading transform.

Run from the Julia repo root against a checkout of the Python reference:

    PYTHONPATH=../skaters/src python parity/gen_transform_vectors.py

Writes parity/transform_vectors.json. test/transform_parity.jl replays the
same-named scenarios and compares the seven probes of parity/vectors.json
(mean, std, logpdf, cdf, two quantiles, crps).
"""
import json
import math
import os
import random

from skaters.api import flesaker, laplace
from skaters.conjugate import conjugate
from skaters.ensemble import precision_weighted_ensemble
from skaters.leaf import leaf
from skaters.transform import (ema_transform, gaussianize, log_transform, power_transform,
                               seasonal_anchor, seasonal_difference, seasonal_scale,
                               standardize, yeo_johnson)

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = 0.3
Q_LO = 0.1
Q_HI = 0.9
BURN = 10
LONG_FROM = 515      # laplace's splice is live from about step 510
LONG_EVERY = 20


def clean(x):
    if x != x:
        return "nan"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return x


def probe_dist(d):
    return [clean(v) for v in
            (d.mean, d.std, d.logpdf(PROBE), d.cdf(PROBE),
             d.quantile(Q_LO), d.quantile(Q_HI), d.crps(PROBE))]


def long_series(n=700, seed=5):
    rng = random.Random(seed)
    x, out = 0.0, []
    for t in range(n):
        x = 0.9 * x + rng.gauss(0.0, 1.0) * (1.0 + 0.5 * math.sin(t / 7.0))
        out.append(x)
    return out


def short_scenarios(lo, hi):
    s = []
    for k in (1, 3):
        suf = "" if k == 1 else f"_k{k}"
        block = [
            ("seasonal_scale", k, conjugate(leaf(k=k), seasonal_scale(7), k=k)),
            ("seasonal_scale_center", k, conjugate(leaf(k=k), seasonal_scale(7, center=True), k=k)),
            ("seasonal_scale_over_sdiff", k,
             conjugate(conjugate(leaf(k=k), seasonal_scale(7), k=k), seasonal_difference(7), k=k)),
            ("gaussianize", k, conjugate(leaf(k=k), gaussianize(), k=k)),
            ("gaussianize_cap", k, conjugate(leaf(k=k), gaussianize(slope_cap=2.0), k=k)),
            ("gaussianize_smooth", k, conjugate(leaf(k=k), gaussianize(smooth=True), k=k)),
            ("gaussianize_delete", k, conjugate(leaf(k=k), gaussianize(guard="delete", slope_cap=2.0), k=k)),
            ("gaussianize_emin", k, conjugate(leaf(k=k), gaussianize(guard="emin", slope_cap=2.0), k=k)),
            ("power_exact", k, conjugate(leaf(k=k), power_transform(0.5, exact=True), k=k)),
            ("yeojohnson_log_exact", k, conjugate(leaf(k=k), yeo_johnson(0.0, exact=True), k=k)),
            ("yeojohnson_half_exact", k, conjugate(leaf(k=k), yeo_johnson(0.5, exact=True), k=k)),
            ("stack_exact_delta", k,
             conjugate(conjugate(leaf(k=k), yeo_johnson(0.5, exact=True), k=k), power_transform(0.5), k=k)),
            ("stack_exact_exact", k,
             conjugate(conjugate(leaf(k=k), yeo_johnson(0.5, exact=True), k=k),
                       power_transform(0.5, exact=True), k=k)),
            ("combine_pushforwards", k, precision_weighted_ensemble(
                [conjugate(leaf(k=k), yeo_johnson(0.5, exact=True), k=k),
                 conjugate(leaf(k=k), power_transform(0.5, exact=True), k=k)], k=k)),
            ("log_lower", k, conjugate(leaf(k=k), log_transform(lower=lo), k=k)),
            ("log_upper", k, conjugate(leaf(k=k), log_transform(upper=hi), k=k)),
            # affine transforms around a pushforward predictive
            ("std_yeojohnson_exact", k,
             conjugate(conjugate(leaf(k=k), yeo_johnson(0.0, exact=True), k=k), standardize(), k=k)),
            ("std_log", k,
             conjugate(conjugate(leaf(k=k), log_transform(lower=-10.0), k=k), standardize(), k=k)),
            ("sscale_log", k,
             conjugate(conjugate(leaf(k=k), log_transform(lower=-10.0), k=k),
                       seasonal_scale(7, center=True), k=k)),
        ]
        s += [(n + suf, kk, f) for n, kk, f in block]
    return s


def long_scenarios(lo, hi):
    s = [
        ("laplace_power", 1, conjugate(laplace(k=1), power_transform(0.5), k=1)),
        ("laplace_yeojohnson", 1, conjugate(laplace(k=1), yeo_johnson(0.5), k=1)),
        ("laplace_gaussianize", 1, conjugate(laplace(k=1), gaussianize(), k=1)),
        ("laplace_stack", 1,
         conjugate(conjugate(laplace(k=1), yeo_johnson(0.5), k=1), power_transform(0.5), k=1)),
        ("laplace_seasonal_scale", 1, conjugate(laplace(k=1), seasonal_scale(7), k=1)),
        ("laplace_ema_k3", 3, conjugate(laplace(k=3), ema_transform(0.05), k=3)),
        ("laplace_sdiff_k3", 3, conjugate(laplace(k=3), seasonal_difference(2), k=3)),
        ("laplace_sanchor_k3", 3, conjugate(laplace(k=3), seasonal_anchor(2), k=3)),
        ("flesaker", 1, flesaker(k=1, lower=lo)),
        ("flesaker_upper_k3", 3, flesaker(k=3, upper=hi)),
        ("laplace_std_yeojohnson_exact", 1,
         conjugate(conjugate(laplace(k=1), yeo_johnson(0.0, exact=True), k=1), standardize(0.05), k=1)),
    ]
    return s


def fixed_spliced(long, short, llo):
    """Exact SplicedDist parameters captured from laplace on the long series,
    rebuilt from those parameters, and pushed through every inverse that
    reads or emits a non-mixture predictive. Both sides rebuild the same
    object, so these probes isolate the transforms from laplace's long-run
    state (see test/transform_parity.jl for the tolerances)."""
    from skaters.dist import Dist
    from skaters.tails import SplicedDist
    from skaters.transform import _widen

    def state_after(tr, series):
        fwd, inv = tr
        st = None
        for y in series:
            _, st = fwd(y, st)
        return inv, st

    f, st, caps = laplace(k=1), None, []
    for i, y in enumerate(long):
        d, st = f(y, st)
        if i >= LONG_FROM and (i - LONG_FROM) % 40 == 0:
            d = d[0]
            assert isinstance(d, SplicedDist)
            caps.append({"comps": [list(c) for c in d.body.components],
                         "par": [d.t_lo, d.t_up, d.zeta_lo, d.zeta_up,
                                 d.g_lo, d.s_lo, d.g_up, d.s_up]})
    g_inv, g_st = state_after(gaussianize(), short)
    s_inv, s_st = state_after(seasonal_scale(7, center=True), short)
    out = []
    for c in caps:
        d = SplicedDist(Dist([tuple(x) for x in c["comps"]]), *c["par"])
        pw = power_transform(0.5)[1]
        pwx = power_transform(0.5, exact=True)[1]
        yj = yeo_johnson(0.5)[1]
        yjx = yeo_johnson(0.0, exact=True)[1]
        lg = log_transform(lower=llo)[1]
        lgu = log_transform(upper=-llo)[1]
        a, b = yj([d], {})[0], pwx([d], {})[0]
        dists = {
            "power": pw([d], {})[0],
            "power_exact": b,
            "yeojohnson": a,
            "yeojohnson_log_exact": yjx([d], {})[0],
            "stack": pw([a], {})[0],
            "stack_exact": pwx([a], {})[0],
            "log_lower": lg([d], {})[0],
            "log_upper": lgu([d], {})[0],
            "gaussianize": g_inv([d], g_st)[0],
            "seasonal_scale": s_inv([d], s_st)[0],
            "widen": _widen(d, 0.7, 0.3),
            "widen_pushforward": _widen(a, 0.7, 0.3),
            "combine": Dist.combine([a, b], [0.3, 0.7]),
            "affine_power_exact": b.affine(2.5, -1.0),
            "affine_power_exact_neg": b.affine(-1.7, 0.5),
            "affine_log_lower_neg": lg([d], {})[0].affine(-0.3, 4.0),
            "shift_yeojohnson_log_exact": yjx([d], {})[0].shift(1.5),
            "scale_log_upper": lgu([d], {})[0].scale(-2.0),
        }
        c["out"] = {k: probe_dist(v) for k, v in dists.items()}
        out.append(c)
    return out


def run(skater, series, keep):
    st, out = None, []
    for i, y in enumerate(series):
        dists, st = skater(y, st)
        if keep(i):
            out.append([i, [probe_dist(d) for d in dists]])
    return out


def main():
    with open(os.path.join(HERE, "vectors.json")) as f:
        short = json.load(f)["series"]
    lo, hi = min(short) - 1.0, max(short) + 1.0
    long = long_series()
    llo, lhi = min(long) - 1.0, max(long) + 1.0
    vec = {"short_bounds": [lo, hi], "long_series": long, "long_bounds": [llo, lhi],
           "probe": PROBE, "q_lo": Q_LO, "q_hi": Q_HI, "scenarios": {}}
    for name, k, f in short_scenarios(lo, hi):
        vec["scenarios"][name] = {"series": "short", "k": k,
                                  "out": run(f, short, lambda i: i >= BURN)}
        print("ok", name)
    for name, k, f in long_scenarios(llo, lhi):
        vec["scenarios"][name] = {"series": "long", "k": k,
                                  "out": run(f, long, lambda i: i >= LONG_FROM and
                                             (i - LONG_FROM) % LONG_EVERY == 0)}
        print("ok", name)
    vec["fixed_spliced"] = fixed_spliced(long, short, llo)
    print("ok fixed_spliced")
    with open(os.path.join(HERE, "transform_vectors.json"), "w") as f:
        json.dump(vec, f, allow_nan=False)


if __name__ == "__main__":
    main()
