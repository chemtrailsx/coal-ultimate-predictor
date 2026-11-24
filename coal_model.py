"""Proximate <-> ultimate analysis of coal: inference with the trained neural-network ensembles.

The weights live in model/coal_model.json (written by the notebook). Only numpy is needed here,
so the app does not depend on any scikit-learn version.

Method (dry, ash-free "daf" route):
    organic matter      Org   = 100 - M - Ash            (any basis: as-received, air-dried, dry)
    inputs to network   VM_daf = 100 * VM / Org,  Ash/Org,  M/Org
    network predicts    C_daf, H_daf, N_daf, S_daf
    oxygen              O_daf = 100 - C_daf - H_daf - N_daf - S_daf      (by difference)
    back to the input basis:  X = X_daf * Org / 100
so that  C + H + N + O + S + M + Ash = 100  holds exactly.

Reverse (ultimate -> proximate):
    Org = C + H + N + O + S,   M + Ash = 100 - Org   (exact)
    network 1:  C,H,N,O,S (daf)                 -> VM_daf      VM = VM_daf * Org / 100,  FC = Org - VM
    network 2:  C,H,N,O,S (daf), (M+Ash)/Org    -> M/Org       (only if moisture/ash is not given)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "coal_model.json")
ELEMENTS = ["C", "H", "N", "O", "S"]
PROXIMATE = ["M", "VM", "Ash", "FC"]


def _relu(x):
    return np.maximum(x, 0.0)


class _Net:
    """One sub-model: input scaler -> several MLPs (and optionally a quadratic regression) -> output scaler."""

    def __init__(self, spec: dict):
        self.features = spec["features"]
        self.x_mean, self.x_scale = np.array(spec["x_mean"]), np.array(spec["x_scale"])
        self.y_mean, self.y_scale = np.array(spec["y_mean"]), np.array(spec["y_scale"])
        self.mlps = [([np.array(W) for W in m["W"]], [np.array(b) for b in m["b"]]) for m in spec["mlps"]]
        self.poly = spec.get("poly")
        if self.poly:
            self.poly_coef = np.array(self.poly["coef"])
            self.poly_intercept = np.array(self.poly["intercept"])
            self.poly_weight = float(self.poly["weight"])

    def _mlp(self, Xs, Ws, bs):
        h = Xs
        for i, (W, b) in enumerate(zip(Ws, bs)):
            h = h @ W + b
            if i < len(Ws) - 1:
                h = _relu(h)
        return h

    @staticmethod
    def _poly2(Xs):
        """Same column order as sklearn PolynomialFeatures(degree=2)."""
        n, k = Xs.shape
        cols = [np.ones(n)] + [Xs[:, i] for i in range(k)]
        cols += [Xs[:, i] * Xs[:, j] for i in range(k) for j in range(i, k)]
        return np.column_stack(cols)

    def predict_daf(self, F: np.ndarray) -> np.ndarray:
        Xs = (F - self.x_mean) / self.x_scale
        nn = np.mean([self._mlp(Xs, W, b) for W, b in self.mlps], axis=0) * self.y_scale + self.y_mean
        if not self.poly:
            return nn
        pr = self._poly2(Xs) @ self.poly_coef.T + self.poly_intercept
        return (1 - self.poly_weight) * nn + self.poly_weight * pr


@dataclass
class Result:
    basis_values: dict   # C, H, N, O, S on the same basis as the input proximate analysis
    dry: dict            # dry basis
    daf: dict            # dry, ash-free basis
    fixed_carbon: float
    organic: float
    warnings: list


class CoalModel:
    def __init__(self, path: str = MODEL_PATH):
        with open(path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.nets = {k: _Net(v) for k, v in self.meta["models"].items()}
        self.rev = {k: _Net(self.meta["reverse"][k]) for k in ("vm", "moist")}

    @staticmethod
    def features(M, VM, Ash, use_moisture=True):
        M, VM, Ash = (np.atleast_1d(np.asarray(v, float)) for v in (M, VM, Ash))
        org = 100.0 - M - Ash
        cols = [100.0 * VM / org, Ash / org] + ([M / org] if use_moisture else [])
        return np.column_stack(cols), org

    def predict_arrays(self, M, VM, Ash, S=None, moisture_known=True):
        """Vectorised prediction. Returns (C,H,N,O,S) on the input basis, shape (n, 5).

        M, VM, Ash : proximate analysis, wt %, all on the same basis.
        S          : optional measured total sulfur on that basis (NaN = not measured).
        moisture_known=False uses the sub-model trained without moisture (for dry-basis data).
        """
        key = "with_moisture" if moisture_known else "no_moisture"
        F, org = self.features(M, VM, Ash, use_moisture=moisture_known)
        P = np.clip(self.nets[key].predict_daf(F), 0.0, None)          # C, H, N, S  (daf)
        if S is not None:
            S = np.atleast_1d(np.asarray(S, float))
            known = ~np.isnan(S)
            P[known, 3] = 100.0 * S[known] / org[known]
            # a measured S is kept exactly: if C+H+N+S would exceed 100 (daf), scale C, H, N down instead
            chn = P[:, :3].sum(axis=1)
            over = known & (chn + P[:, 3] > 100.0)
            P[over, :3] *= ((100.0 - P[over, 3]) / chn[over])[:, None]
        O = np.clip(100.0 - P.sum(axis=1), 0.0, None)
        daf = np.column_stack([P[:, 0], P[:, 1], P[:, 2], O, P[:, 3]])
        daf = 100.0 * daf / daf.sum(axis=1, keepdims=True)              # exact closure
        return daf * org[:, None] / 100.0

    def predict(self, M, VM, Ash, S=None, basis="as-received", moisture_for_dry=None) -> Result:
        """Single sample. basis: 'as-received', 'air-dried' or 'dry'."""
        warn = []
        M = 0.0 if basis == "dry" else float(M)
        VM, Ash = float(VM), float(Ash)
        org = 100.0 - M - Ash
        fc = org - VM
        if min(M, VM, Ash) < 0:
            raise ValueError("Values cannot be negative.")
        if fc < 0:
            raise ValueError("Moisture + volatile matter + ash is more than 100 %.")
        if VM <= 0:
            raise ValueError("Volatile matter must be above zero.")
        if S is not None and S >= org:
            raise ValueError("Sulphur cannot be larger than the organic matter.")
        if org < 15:
            warn.append("Less than 15 % organic matter (100 - M - Ash): this is closer to carbonaceous shale "
                        "than coal and is outside the training range.")
        r = self.meta["ranges"]
        vmdaf, aorg = 100 * VM / org, Ash / org
        if not r["VMdaf"][0] <= vmdaf <= r["VMdaf"][1]:
            warn.append(f"Volatile matter on a dry, ash-free basis is {vmdaf:.1f} %, outside the training range "
                        f"({r['VMdaf'][0]:.0f}-{r['VMdaf'][1]:.0f} %). Treat the result with caution.")
        if aorg > r["A_org"][1]:
            warn.append("Ash is higher than in any training sample relative to the organic matter.")

        if basis == "dry":
            if moisture_for_dry is not None and moisture_for_dry > 0:
                # convert dry values to the as-received basis given the moisture, predict, convert back
                m = float(moisture_for_dry)
                f = (100 - m) / 100
                out = self.predict_arrays(m, VM * f, Ash * f, None if S is None else S * f)[0] / f
            else:
                warn.append("No moisture given: the model without moisture is used, which is less accurate "
                            "(moisture helps it recognise coal rank).")
                out = self.predict_arrays(0.0, VM, Ash, S, moisture_known=False)[0]
        else:
            if M / org > r["M_org"][1]:
                warn.append("Moisture is higher than in any training sample relative to the organic matter.")
            out = self.predict_arrays(M, VM, Ash, S)[0]

        vals = dict(zip(ELEMENTS, out))
        dry_f = 100.0 / (100.0 - M)
        dry = {k: v * dry_f for k, v in vals.items()}
        daf = {k: 100.0 * v / org for k, v in vals.items()}
        return Result(vals, dry, daf, fc, org, warn)




# ====================================================================== ultimate -> proximate
@dataclass
class ReverseResult:
    basis_values: dict   # M, VM, Ash, FC on the input basis (None where not determinable, daf input)
    dry: dict            # dry basis (None where not determinable)
    daf: dict            # VM and FC on a dry, ash-free basis
    moisture_source: str  # 'given', 'from given ash', 'predicted', 'zero (dry basis)', 'not determinable'
    organic: float
    warnings: list


def _rev_features(C, H, N, O, S):
    U = np.column_stack([np.atleast_1d(np.asarray(v, float)) for v in (C, H, N, O, S)])
    org = U.sum(axis=1)
    return 100.0 * U / org[:, None], org


def _rev_vm_daf(model, D):
    return np.clip(model.rev["vm"].predict_daf(D)[:, 0], 0.0, 100.0)


def _rev_m_org(model, D, org):
    F = np.column_stack([D, (100.0 - org) / org])
    return model.rev["moist"].predict_daf(F)[:, 0]


def predict_proximate_arrays(model, C, H, N, O, S, M=None):
    """Vectorised, as-received/air-dried basis. M: optional measured moisture (NaN = unknown).
    Returns array (n, 4): M, VM, Ash, FC."""
    D, org = _rev_features(C, H, N, O, S)
    rest = np.clip(100.0 - org, 0.0, None)
    Mp = np.clip(_rev_m_org(model, D, org) * org, 0.0, rest)
    if M is not None:
        M = np.atleast_1d(np.asarray(M, float))
        known = ~np.isnan(M)
        Mp[known] = np.clip(M[known], 0.0, rest[known])
    VM = _rev_vm_daf(model, D) * org / 100.0
    return np.column_stack([Mp, VM, rest - Mp, org - VM])


def predict_proximate(model, C, H, N, O, S, basis="as-received", M=None, Ash=None) -> ReverseResult:
    """Single sample. basis: 'as-received', 'air-dried', 'dry' or 'daf'.
    M, Ash: optional measured values on the same basis (for 'daf': the as-received M and Ash of the sample)."""
    vals = [float(v) for v in (C, H, N, O, S)]
    if min(vals) < 0:
        raise ValueError("Values cannot be negative.")
    if vals[0] <= 0 or vals[1] <= 0:
        raise ValueError("Carbon and hydrogen must be above zero.")
    for name, v in (("Moisture", M), ("Ash", Ash)):
        if v is not None and v < 0:
            raise ValueError(f"{name} cannot be negative.")
    tot = sum(vals)
    warn = []
    if basis == "daf":
        if abs(tot - 100.0) > 1.0:
            raise ValueError(f"On a dry, ash-free basis C + H + N + O + S should be 100 %, but it is {tot:.2f} %.")
        vals = [100.0 * v / tot for v in vals]
        if M is not None and Ash is not None:
            org = 100.0 - M - Ash
            if org <= 0:
                raise ValueError("Moisture + ash must be less than 100 %.")
            vals = [v * org / 100.0 for v in vals]
            basis_eff, Mv, Av, src = "as-received", float(M), float(Ash), "given"
        else:
            basis_eff, Mv, Av, src = "daf", None, None, "not determinable"
            org = 100.0
    else:
        if tot > 100.0 + 1e-9:
            raise ValueError(f"C + H + N + O + S is {tot:.2f} %, more than 100 %. Check the values and the basis.")
        org = tot
        rest = 100.0 - org
        basis_eff = basis
        if basis == "dry":
            Mv, Av, src = 0.0, rest, "zero (dry basis)"
        elif M is not None:
            if M > rest + 1e-9:
                raise ValueError(f"Moisture cannot exceed 100 - (C+H+N+O+S) = {rest:.2f} %.")
            Mv, Av, src = float(M), rest - float(M), "given"
        elif Ash is not None:
            if Ash > rest + 1e-9:
                raise ValueError(f"Ash cannot exceed 100 - (C+H+N+O+S) = {rest:.2f} %.")
            Mv, Av, src = rest - float(Ash), float(Ash), "from given ash"
        else:
            D, o = _rev_features(*vals)
            Mv = float(np.clip(_rev_m_org(model, D, o)[0] * org, 0.0, rest))
            Av, src = rest - Mv, "predicted"
    if org < 15:
        warn.append("Less than 15 % organic matter (C + H + N + O + S): this is closer to carbonaceous shale than "
                    "coal and is outside the training range.")
    D, _ = _rev_features(*vals)
    r = model.meta["reverse"]["ranges"]
    if not r["Cdaf"][0] <= D[0, 0] <= r["Cdaf"][1]:
        warn.append(f"Carbon on a dry, ash-free basis is {D[0, 0]:.1f} %, outside the training range "
                    f"({r['Cdaf'][0]:.0f}-{r['Cdaf'][1]:.0f} %). Treat the result with caution.")
    if not r["Odaf"][0] <= D[0, 3] <= r["Odaf"][1]:
        warn.append(f"Oxygen on a dry, ash-free basis is {D[0, 3]:.1f} %, outside the training range "
                    f"({r['Odaf'][0]:.0f}-{r['Odaf'][1]:.0f} %).")
    vm_daf = float(_rev_vm_daf(model, D)[0])
    daf = {"VM": vm_daf, "FC": 100.0 - vm_daf}
    if Mv is None:
        return ReverseResult({k: None for k in PROXIMATE}, {k: None for k in PROXIMATE}, daf, src, org, warn)
    org_b = 100.0 - Mv - Av
    VM = vm_daf * org_b / 100.0
    vals_b = {"M": Mv, "VM": VM, "Ash": Av, "FC": org_b - VM}
    f = 100.0 / (100.0 - Mv)
    dry = {"M": 0.0, "VM": VM * f, "Ash": Av * f, "FC": (org_b - VM) * f}
    return ReverseResult(vals_b, dry, daf, src, org_b, warn)


CoalModel.predict_proximate = predict_proximate
CoalModel.predict_proximate_arrays = predict_proximate_arrays


if __name__ == "__main__":
    m = CoalModel()
    r = m.predict(M=6.0, VM=24.0, Ash=35.0)
    print("prox -> ult:", {k: round(v, 2) for k, v in r.basis_values.items()}, r.warnings)
    q = m.predict_proximate(47.77, 3.04, 1.00, 6.51, 0.68)
    print("ult -> prox:", {k: round(v, 2) for k, v in q.basis_values.items()}, q.moisture_source, q.warnings)
