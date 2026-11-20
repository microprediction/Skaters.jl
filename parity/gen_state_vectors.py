"""Cross-language resume vectors: Python checkpoints in the canonical state
layout (skaters docs/STATE.md) that Julia must load and continue.

For every scenario below, the skater runs over its series; at t = 60, 250
and 530 (when 20 more steps fit) the state is written with state_to_json,
together with the probes of the next 20 Python outputs. test/state_resume.jl
loads each checkpoint with state_from_json, steps 20, and must match Python
to 1e-6, and match a Julia run from scratch.

The scenarios mirror the reference's tests/test_state_json.py for every
skater this port has. The reference's mahalanobis, zbank and gpdtail are
leaving skaters (skaters#90); conforming_leaf and ensemble_leaf are not in
the reference's public API.

Run from the Julia repo root against a checkout of the Python reference:

    PYTHONPATH=../skaters/src python parity/gen_state_vectors.py

Writes parity/state_vectors.json.
"""
import json
import math
import os
import random
import sys

from skaters import (bayesian_ensemble, conjugate, dantzig, ema, flesaker, granger,
                     homogenize, laplace, leaf, multiscale, parade, period_detector,
                     precision_weighted_ensemble, residual_transform, search, state_to_json,
                     sticky, terminal_leaf_ensemble)
from skaters import transform as T
from skaters.leaf import crps_leaf, garch_leaf, scale_mixture_leaf
from skaters.tails import gpdtails

HERE = os.path.dirname(os.path.abspath(__file__))
N = 620
CHECKPOINTS = (60, 250, 530)
CONTINUE = 20
PROBE = 0.3
Q_LO = 0.1
Q_HI = 0.9


def _gen_vectors():
    """The reference's parity/gen_vectors.py, beside its src/."""
    import skaters
    ref = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(skaters.__file__))))
    sys.path.insert(0, os.path.join(ref, "parity"))
    import gen_vectors
    return gen_vectors


def _c(t, k=1):
    return conjugate(leaf(k=k), t, k=k)


def transforms():
    return [
        ("difference", T.difference()),
        ("fractional_difference", T.fractional_difference(0.4, 30)),
        ("standardize", T.standardize()),
        ("ema_transform", T.ema_transform(0.1)),
        ("ou_transform", T.ou_transform(0.1)),
        ("theta", T.theta(0.1)),
        ("drift", T.drift(0.05, 0.01)),
        ("holt_linear", T.holt_linear(0.1, 0.05)),
        ("garch", T.garch()),
        ("seasonal_difference", T.seasonal_difference(7)),
        ("seasonal_anchor", T.seasonal_anchor(7)),
        ("seasonal_scale", T.seasonal_scale(7)),
        ("seasonal_scale_center", T.seasonal_scale(7, center=True)),
        ("power_transform", T.power_transform(0.5)),
        ("power_transform_exact", T.power_transform(0.5, exact=True)),
        ("yeo_johnson", T.yeo_johnson(0.5)),
        ("yeo_johnson_exact", T.yeo_johnson(0.5, exact=True)),
        ("ar", T.ar(2)),
        ("grouped_ar", T.grouped_ar(8)),
        ("gaussianize", T.gaussianize()),
        ("gaussianize_smooth", T.gaussianize(smooth=True)),
    ]


def scenarios(series, repeat, gaps):
    """(name, skater, series name). Names must match test/state_resume.jl."""
    short = "short"
    out = []
    for name, t in transforms():
        out.append(("conjugate_" + name, _c(t), "positive" if name.startswith("power") else "series"))
    out += [
        ("conjugate_log_transform", conjugate(leaf(k=1), T.log_transform(), k=1), "positive"),
        ("laplace", laplace(1), "series"),
        ("laplace_k3", laplace(3), "series"),
        ("laplace_likelihood", laplace(1, objective="likelihood"), "series"),
        ("laplace_gaussian_tails", laplace(1, tails="gaussian"), short),
        ("laplace_repeat", laplace(1), "repeat"),
        ("laplace_gaps", laplace(1), "gaps"),
        ("laplace_k3_gaps", laplace(3), "gaps"),
        ("granger", granger(1), "series"),
        ("flesaker", flesaker(1, lower=-100.0), "series"),
        ("dantzig", dantzig(1, expand_interval=50), short),
        ("search_k2", search(k=2, expand_interval=40, max_pool=12), short),
        ("multiscale_k4", multiscale(lambda kk: _c(T.ema_transform(0.1), kk), 4), short),
        ("parade_spliced", parade(gpdtails(_c(T.ema_transform(0.1)), k=1, nexc=50, warmup=100), 1), "series"),
        ("parade_laplace_k3", parade(laplace(3), 3), "series"),
        ("sticky_ema_repeat", sticky(_c(T.ema_transform(0.1)), k=1), "repeat"),
        ("homogenize", homogenize(_c(T.ema_transform(0.1)), k=1), short),
        ("homogenize_k2_weights", homogenize(_c(T.ema_transform(0.1), 2), k=2, couple="weights"), short),
        ("homogenize_k2_shared", homogenize(_c(T.ema_transform(0.1), 2), k=2, couple="shared"), short),
    ]
    for model in ("m0", "m1", "m2"):
        out.append((f"residual_transform_{model}",
                    residual_transform(_c(T.ema_transform(0.1)), model=model), short))
    out += [
        ("pw_ensemble_k2", precision_weighted_ensemble([ema(0.05, k=2), ema(0.2, k=2)], k=2), short),
        ("bayes_ensemble_k2", bayesian_ensemble([ema(0.05, k=2), _c(T.difference(), 2)], k=2), short),
        ("terminal_adaptive", terminal_leaf_ensemble(
            [ema(0.05, k=1), _c(T.difference())], k=1, depths=[0, 1], adaptive_temperature=True), short),
        ("terminal_k3", terminal_leaf_ensemble(
            [ema(0.05, k=3), _c(T.difference(), 3)], k=3, depths=[0, 1], forget=0.99), short),
        ("scale_mixture_leaf", scale_mixture_leaf(k=1), "series"),
        ("crps_leaf", crps_leaf(k=1), "series"),
        ("garch_leaf", garch_leaf(k=1), "series"),
    ]
    return out


def clean(x):
    if x != x:
        return "nan"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return x


def probe(d):
    return [clean(v) for v in (d.mean, d.std, d.logpdf(PROBE), d.cdf(PROBE),
                               d.quantile(Q_LO), d.quantile(Q_HI), d.crps(PROBE))]


def series_table():
    gv = _gen_vectors()
    series = gv.make_series(n=N, seed=12345)
    return {"series": series, "short": series[:300],
            "positive": [abs(v) + 1.0 for v in series],
            "repeat": gv.make_repeat_series(n=N, seed=99),
            "gaps": gv.make_gap_series(n=N, seed=4242)}


def y_of(v):
    return v


def main():
    tab = series_table()
    vec = {"probe": PROBE, "q_lo": Q_LO, "q_hi": Q_HI, "continue": CONTINUE,
           "series": {k: [clean(v) for v in vs] for k, vs in tab.items()}, "scenarios": {}}
    only = set(sys.argv[1:])
    for name, sk, sname in scenarios(tab["series"], tab["repeat"], tab["gaps"]):
        if only and name not in only:
            continue
        ys = tab[sname]
        n = len(ys)
        cps = [c for c in CHECKPOINTS if c + CONTINUE < n] or [n // 2]
        st, rows = None, []
        for t in range(n):
            _, st = sk(y_of(ys[t]), st)
            if t in cps:
                plain = json.loads(json.dumps(state_to_json(st), allow_nan=False))
                # the continuation runs on a restored copy, so the outputs are
                # exactly what a resumed Python stream produces
                from skaters import state_from_json
                twin = state_from_json(plain)
                outs = []
                for m in range(1, CONTINUE + 1):
                    d, twin = sk(y_of(ys[t + m]), twin)
                    outs.append([probe(x) for x in d])
                rows.append({"t": t, "state": plain, "next": outs})
        vec["scenarios"][name] = {"series": sname, "checkpoints": rows}
        print("ok", name, flush=True)
    with open(os.path.join(HERE, "state_vectors.json"), "w") as f:
        json.dump(vec, f, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
