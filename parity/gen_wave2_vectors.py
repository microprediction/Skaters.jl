"""Parity vectors for the rest of the reference's public API: homogenize,
residual_transform (with ResidualDiagnostics), dantzig / search, and the spec
system's build, spec_name, to_json and from_json.

Run from the Julia repo root against a checkout of the Python reference:

    PYTHONPATH=../skaters/src python parity/gen_wave2_vectors.py

Writes parity/wave2_vectors.json; test/wave2_parity.jl replays it.
"""
import json
import math
import os
import random

from skaters import (build, conjugate, dantzig, ema_transform, from_json, homogenize,
                     homogenize_candidates, laplace, leaf, residual_transform, search,
                     spec_name, to_json, ResidualDiagnostics)
from skaters.spec import (conjugate_spec, diff_spec, ema_spec, ensemble_spec, frac_spec,
                          leaf_spec, std_spec, ema_t_spec)

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = 0.3
Q_LO = 0.1
Q_HI = 0.9
BURN = 10
LONG_FROM = 515
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


SPECS = [
    leaf_spec(1),
    ema_spec(0.05, 1),
    ensemble_spec(ema_spec(0.01, 1), ema_spec(0.1, 1), k=1),
    conjugate_spec(ensemble_spec(ema_spec(0.01, 1), ema_spec(0.1, 1), k=1), diff_spec()),
    conjugate_spec(leaf_spec(3), frac_spec(0.35, 30)),
    conjugate_spec(conjugate_spec(leaf_spec(1), std_spec(0.02)), ema_t_spec(1e-05)),
    conjugate_spec(ema_spec(0.2, 3), frac_spec(0.4)),
]


def small_candidates():
    return homogenize_candidates(rho_grid=(0.0, 0.9), gain_grid=(0.1,), delta_grid=(0.0, 0.5))


def short_scenarios():
    s = [
        ("rt_m0", 1, residual_transform(leaf(k=1), model="m0")),
        ("rt_m1", 1, residual_transform(leaf(k=1), model="m1")),
        ("rt_m2", 1, residual_transform(leaf(k=1), model="m2")),
        ("rt_m2_ema", 1, residual_transform(conjugate(leaf(k=1), ema_transform(0.1), k=1),
                                            model="m2", lr=0.05)),
        ("rt_laplace", 1, residual_transform(laplace(k=1))),
        ("hg", 1, homogenize(leaf(k=1))),
        ("hg_small", 1, homogenize(leaf(k=1), candidates=small_candidates())),
        ("hg_laplace", 1, homogenize(laplace(k=1))),
        ("dantzig", 1, dantzig(k=1, expand_interval=50)),
        ("search_k3", 3, search(k=3, expand_interval=40)),
    ]
    for couple in ("none", "weights", "state", "shared"):
        s.append((f"hg_k3_{couple}", 3, homogenize(leaf(k=3), k=3, couple=couple, lam=0.4)))
    for i, sp in enumerate(SPECS):
        s.append((f"spec_{i}", _k(sp), build(sp)))
    return s


def _k(sp):
    if "k" in sp:
        return sp["k"]
    if "skater" in sp:
        return _k(sp["skater"])
    return _k(sp["skaters"][0])


def run(skater, series, keep, zs=None):
    st, out = None, []
    for i, y in enumerate(series):
        dists, st = skater(y, st)
        if zs is not None:
            zs.append(st["z"])
        if keep(i):
            out.append([i, [probe_dist(d) for d in dists]])
    return out


def main():
    with open(os.path.join(HERE, "vectors.json")) as f:
        short = json.load(f)["series"]
    long = long_series()
    vec = {"long_series": long, "probe": PROBE, "q_lo": Q_LO, "q_hi": Q_HI, "scenarios": {}}
    zstream = []
    for name, k, f in short_scenarios():
        zs = zstream if name == "rt_m2" else None
        vec["scenarios"][name] = {"series": "short", "k": k,
                                  "out": run(f, short, lambda i: i >= BURN, zs)}
        print("ok", name)
    for name, k, f in [("hg_laplace_long", 1, homogenize(laplace(k=1))),
                       ("rt_laplace_long", 1, residual_transform(laplace(k=1))),
                       ("hg_laplace_k3_long", 3, homogenize(laplace(k=3), k=3, couple="weights"))]:
        vec["scenarios"][name] = {"series": "long", "k": k,
                                  "out": run(f, long, lambda i: i >= LONG_FROM and
                                             (i - LONG_FROM) % LONG_EVERY == 0)}
        print("ok", name)
    # residual_transform's z stream and the diagnostics fed from it
    vec["rt_m2_z"] = [clean(z) if z is not None else None for z in zstream]
    rd = ResidualDiagnostics(K=3, halflife=20.0)
    for z in zstream:
        if z is not None:
            rd.update(z)
    vec["diagnostics"] = rd.summary()
    vec["candidates"] = [dict(c) for c in homogenize_candidates()]
    vec["specs"] = [{"json": to_json(sp), "name": spec_name(sp),
                     "roundtrip": to_json(from_json(to_json(sp)))} for sp in SPECS]
    with open(os.path.join(HERE, "wave2_vectors.json"), "w") as f:
        json.dump(vec, f, allow_nan=False)


if __name__ == "__main__":
    main()
