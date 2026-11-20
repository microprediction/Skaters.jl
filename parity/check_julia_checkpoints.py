"""Resume Julia-written checkpoints in the reference.

Reads the file parity/write_julia_checkpoints.jl wrote; for every scenario
and checkpoint, restores the state with state_from_json, steps 20 in Python,
and compares with Python's own continuation stored in
parity/state_vectors.json, to 1e-6.

    PYTHONPATH=../skaters/src python parity/check_julia_checkpoints.py /tmp/julia_checkpoints.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_state_vectors as G  # noqa: E402
from skaters import assert_plain_state, state_from_json  # noqa: E402


def num(x):
    return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}.get(x, x) if isinstance(x, str) else x


def close(g, w):
    if math.isnan(w):
        return math.isnan(g)
    if math.isinf(w):
        return g == w
    return abs(g - w) <= 1e-6 + 1e-6 * abs(w)


def main(path):
    with open(path) as f:
        jl = json.load(f)
    with open(os.path.join(HERE, "state_vectors.json")) as f:
        vec = json.load(f)
    tab = {k: [num(x) for x in xs] for k, xs in vec["series"].items()}
    fails, checked = 0, 0
    for name, sk, sname in G.scenarios(None, None, None):
        ys = tab[sname]
        want = {c["t"]: c["next"] for c in vec["scenarios"][name]["checkpoints"]}
        bad = 0
        for row in jl[name]:
            t = row["t"]
            st = assert_plain_state(state_from_json(row["state"]))
            for m in range(1, G.CONTINUE + 1):
                d, st = sk(ys[t + m], st)
                got = [G.probe(x) for x in d]
                for h, (gh, wh) in enumerate(zip(got, want[t][m - 1])):
                    for j, (g, w) in enumerate(zip(gh, wh)):
                        checked += 1
                        if not close(num(g), num(w)):
                            bad += 1
                            if fails + bad <= 10:
                                print(f"FAIL {name} t={t} +{m} h{h + 1} probe {j + 1}: {g} vs {w}")
        fails += bad
        print(("ok   " if bad == 0 else "FAIL ") + name, flush=True)
    print(f"{checked} values checked")
    if fails:
        raise SystemExit(f"JULIA CHECKPOINTS IN PYTHON FAILED: {fails}")
    print("JULIA CHECKPOINTS RESUME IN PYTHON OK")


if __name__ == "__main__":
    main(sys.argv[1])
