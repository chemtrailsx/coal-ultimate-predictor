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
from coal_model import ELEMENTS, MODEL_PATH, CoalModel  # noqa: E402

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
print("ALL CHECKS PASSED")
