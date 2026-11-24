"""Checks that the deployed predictors agree with each other and obey the physics.

1. Python inference (coal_model.py, used by the Streamlit app) vs the browser JavaScript (web/index.html).
2. Closure: C + H + N + O + S + M + Ash = 100 on every prediction, no negative values.
3. Python inference reproduces the measured Indian and US samples within the cross-validated error.
Run from the project folder:  python tests/test_consistency.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from coal_model import ELEMENTS, MODEL_PATH, PROXIMATE, CoalModel  # noqa: E402

rng = np.random.default_rng(42)
m = CoalModel()

# ---- random but valid cases on all three bases, with and without measured S
cases = []
for i in range(2000):
    basis = ["as-received", "air-dried", "dry"][i % 3]
    M = 0.0 if basis == "dry" else float(rng.uniform(0.5, 40))
    Ash = float(rng.uniform(1, 55))
    org = 100 - M - Ash
    if org < 16:
        continue
    VM = float(rng.uniform(0.04, 0.6) * org)
    S = float(rng.uniform(0.2, 4)) if i % 4 == 0 else None
    Mdry = float(rng.uniform(1, 30)) if (basis == "dry" and i % 2 == 0) else None
    cases.append(dict(basis=basis, M=M, VM=VM, Ash=Ash, FC=None, S=S, Mdry=Mdry))

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(cases, f)
js = json.loads(subprocess.run(["node", os.path.join(ROOT, "tests", "js_predict.js"), MODEL_PATH, f.name],
                               capture_output=True, text=True, check=True).stdout)
os.unlink(f.name)

max_diff, max_closure = 0.0, 0.0
for c, j in zip(cases, js):
    r = m.predict(c["M"], c["VM"], c["Ash"], S=c["S"], basis=c["basis"], moisture_for_dry=c["Mdry"])
    assert "errors" not in j, (c, j)
    py = np.array([r.basis_values[k] for k in ELEMENTS])
    jv = np.array([j["basis"][k] for k in ELEMENTS])
    max_diff = max(max_diff, float(np.abs(py - jv).max()))
    assert (py >= 0).all(), (c, py)
    closure = py.sum() + c["M"] + c["Ash"]
    max_closure = max(max_closure, abs(closure - 100))
    if c["S"] is not None:
        assert abs(r.basis_values["S"] - c["S"]) < 1e-9
print(f"{len(cases)} cases | max |Python - JavaScript| = {max_diff:.2e} pp | max closure error = {max_closure:.2e} pp")
assert max_diff < 1e-9 and max_closure < 1e-9

# ---- bad input is rejected by both
bad = [dict(basis="as-received", M=30, VM=40, Ash=40, FC=None, S=None, Mdry=None),
       dict(basis="air-dried", M=5, VM=0, Ash=20, FC=None, S=None, Mdry=None),
       dict(basis="air-dried", M=5, VM=25, Ash=20, FC=60, S=None, Mdry=None)]
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(bad, f)
jb = json.loads(subprocess.run(["node", os.path.join(ROOT, "tests", "js_predict.js"), MODEL_PATH, f.name],
                               capture_output=True, text=True, check=True).stdout)
os.unlink(f.name)
assert all("errors" in x for x in jb), jb
for b in bad[:2]:
    try:
        m.predict(b["M"], b["VM"], b["Ash"], basis=b["basis"])
        raise AssertionError("should have failed")
    except ValueError:
        pass
print("invalid inputs rejected by both implementations")

# ---- fit on the training data (the model saw these, so errors must be small)
SHORT = {"Moisture": "M", "Volatile Matter": "VM", "Standard Ash": "Ash", "Carbon": "C", "Hydrogen": "H",
         "Nitrogen": "N", "Oxygen": "O", "Sulfur": "S"}
for name in ("usgs_india_clean.csv", "coalqual_clean.csv"):
    d = pd.read_csv(os.path.join(ROOT, "data", name)).rename(columns=SHORT)
    P = m.predict_arrays(d.M.values, d.VM.values, d.Ash.values)
    mae = np.abs(P - d[ELEMENTS].values).mean(axis=0)
    print(name, "training-fit MAE:", dict(zip(ELEMENTS, mae.round(3))))
    assert mae[0] < 2.0 and mae[1] < 0.3

# ====================================================================== ultimate -> proximate
def run_js(cs):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cs, f)
    res = json.loads(subprocess.run(["node", os.path.join(ROOT, "tests", "js_predict.js"), MODEL_PATH, f.name],
                                    capture_output=True, text=True, check=True).stdout)
    os.unlink(f.name)
    return res

rcases = []
for i in range(2000):
    basis = ["as-received", "air-dried", "dry", "daf"][i % 4]
    rank = rng.uniform(0, 1)                          # lignite .. anthracite
    cdaf = 68 + 26 * rank
    hdaf = rng.uniform(3.0, 5.8) - 2.0 * max(rank - 0.8, 0)
    ndaf, sdaf = rng.uniform(0.6, 2.0), rng.uniform(0.3, 4)
    odaf = 100 - cdaf - hdaf - ndaf - sdaf
    if odaf < 1:
        continue
    org = 100.0 if basis == "daf" else float(rng.uniform(25, 95))
    U = [x * org / 100 for x in (cdaf, hdaf, ndaf, odaf, sdaf)]
    M = A = None
    rest = 100 - org
    if basis in ("as-received", "air-dried"):
        k = i % 3
        if k == 1:
            M = float(rng.uniform(0, rest))
        elif k == 2:
            A = float(rng.uniform(0, rest))
    elif basis == "daf" and i % 2 == 0:
        M = float(rng.uniform(0.5, 30)); A = float(rng.uniform(1, 40))
    rcases.append(dict(dir="u2p", basis=basis, C=U[0], H=U[1], N=U[2], O=U[3], S=U[4], M=M, Ash=A))

rjs = run_js(rcases)
mx, mclose = 0.0, 0.0
for c, j in zip(rcases, rjs):
    q = m.predict_proximate(c["C"], c["H"], c["N"], c["O"], c["S"], basis=c["basis"], M=c["M"], Ash=c["Ash"])
    assert "errors" not in j, (c, j)
    assert j["src"] == q.moisture_source, (c, j["src"], q.moisture_source)
    mx = max(mx, abs(j["daf"]["VM"] - q.daf["VM"]))
    if q.basis_values["M"] is None:
        assert j["basis"] is None
        continue
    pv = np.array([q.basis_values[k] for k in PROXIMATE]); jv = np.array([j["basis"][k] for k in PROXIMATE])
    mx = max(mx, float(np.abs(pv - jv).max()))
    assert (pv >= -1e-9).all(), (c, pv)
    mclose = max(mclose, abs(pv.sum() - 100))
    if c["basis"] != "daf":   # M + Ash must equal 100 - (C+H+N+O+S) exactly
        assert abs(pv[0] + pv[2] - (100 - sum(c[k] for k in ELEMENTS))) < 1e-9
    if c["M"] is not None:
        assert abs(pv[0] - c["M"]) < 1e-9
    if c["Ash"] is not None:
        assert abs(pv[2] - c["Ash"]) < 1e-9
print(f"{len(rcases)} reverse cases | max |Python - JavaScript| = {mx:.2e} pp | max closure error = {mclose:.2e} pp")
assert mx < 1e-9 and mclose < 1e-9

# vectorised batch path == single-sample path
sub = [c for c in rcases if c["basis"] == "as-received"][:200]
B = m.predict_proximate_arrays(*(np.array([c[k] for c in sub]) for k in ELEMENTS),
                               M=np.array([np.nan if c["M"] is None else c["M"] for c in sub]))
for c, b in zip(sub, B):
    if c["Ash"] is not None:
        continue
    q = m.predict_proximate(c["C"], c["H"], c["N"], c["O"], c["S"], M=c["M"])
    assert np.abs(b - [q.basis_values[k] for k in PROXIMATE]).max() < 1e-9
print("batch and single-sample reverse predictions agree")

rbad = [dict(dir="u2p", basis="as-received", C=70, H=5, N=1, O=20, S=10, M=None, Ash=None),     # sum > 100
        dict(dir="u2p", basis="daf", C=60, H=4, N=1, O=10, S=1, M=None, Ash=None),             # daf not 100
        dict(dir="u2p", basis="as-received", C=50, H=3, N=1, O=6, S=1, M=50, Ash=None),       # M > rest
        dict(dir="u2p", basis="as-received", C=0, H=3, N=1, O=6, S=1, M=None, Ash=None)]      # C = 0
assert all("errors" in x for x in run_js(rbad))
for b in rbad:
    try:
        m.predict_proximate(b["C"], b["H"], b["N"], b["O"], b["S"], basis=b["basis"], M=b["M"])
        raise AssertionError("should have failed: %s" % b)
    except ValueError:
        pass
print("invalid reverse inputs rejected by both implementations")

for name in ("usgs_india_clean.csv", "coalqual_clean.csv"):
    d = pd.read_csv(os.path.join(ROOT, "data", name)).rename(columns=dict(SHORT, **{"Fixed Carbon": "FC"}))
    P = m.predict_proximate_arrays(*(d[k].values for k in ELEMENTS))
    mae = np.abs(P - d[PROXIMATE].values).mean(axis=0)
    print(name, "reverse training-fit MAE:", dict(zip(PROXIMATE, mae.round(3))))
    assert mae[1] < 2.5

print("ALL CHECKS PASSED")
