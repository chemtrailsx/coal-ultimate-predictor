"""Coal proximate <-> ultimate analysis predictor (web app).

Proximate -> ultimate: enter moisture, volatile matter, ash (fixed carbon) and get C, H, N, S, O.
Ultimate -> proximate: enter C, H, N, S, O and get moisture, volatile matter, ash, fixed carbon.
Works for US (ASTM, as-received) and Indian (IS 1350, air-dried) reporting conventions.

Run locally:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from coal_model import ELEMENTS, PROXIMATE, CoalModel

NAMES = {"C": "Carbon", "H": "Hydrogen", "N": "Nitrogen", "O": "Oxygen (by difference)", "S": "Sulphur",
         "M": "Moisture", "VM": "Volatile matter", "Ash": "Ash", "FC": "Fixed carbon (by difference)"}
BASES = {"As-received (ARB) — US / ASTM D3172 & D3176 style": "as-received",
         "Air-dried (ADB) — Indian IS 1350 style": "air-dried",
         "Dry (DB)": "dry"}
BASES_U = dict(BASES, **{"Dry, ash-free (DAF)": "daf"})
EXAMPLES = {
    "Indian non-coking coal, high ash (Sohagpur-type, ADB)": dict(basis="air-dried", M=6.0, VM=24.0, Ash=35.0),
    "Indian power-plant coal, G11-type (ADB)": dict(basis="air-dried", M=7.0, VM=22.0, Ash=40.0),
    "Indian coking coal, washed (Jharia-type, ADB)": dict(basis="air-dried", M=1.2, VM=24.0, Ash=17.0),
    "Indian lignite (Rajasthan / Neyveli-type, ARB)": dict(basis="as-received", M=35.0, VM=28.0, Ash=10.0),
    "US Appalachian bituminous (ARB)": dict(basis="as-received", M=2.5, VM=33.0, Ash=9.0),
    "US Powder River Basin sub-bituminous (ARB)": dict(basis="as-received", M=27.0, VM=31.0, Ash=5.5),
    "US Pennsylvania anthracite (ARB)": dict(basis="as-received", M=4.0, VM=5.0, Ash=11.0),
}
EXAMPLES_U = {   # real samples from the training data (USGS-GSI India, USGS COALQUAL)
    "Indian bituminous, Sohagpur AML-1 (ARB)": dict(basis="as-received", C=63.35, H=4.00, N=1.40, O=8.09, S=0.57, M=9.71),
    "Indian high-ash bituminous (ADB)": dict(basis="air-dried", C=47.77, H=3.04, N=1.00, O=6.51, S=0.68, M=None),
    "Indian lignite, Rajasthan (ARB)": dict(basis="as-received", C=39.15, H=2.82, N=0.71, O=10.92, S=1.41, M=None),
    "US bituminous, COALQUAL D160759 (ARB)": dict(basis="as-received", C=78.8, H=4.46, N=1.3, O=3.94, S=0.7, M=4.8),
    "US bituminous, dry, ash-free (DAF)": dict(basis="daf", C=84.5, H=5.48, N=1.67, O=6.78, S=1.57, M=None),
}

st.set_page_config(page_title="Coal Analysis Predictor", page_icon="⛏️", layout="wide")


@st.cache_resource
def load_model():
    return CoalModel()


model = load_model()
meta = model.meta

st.title("⛏️ Coal proximate ⇄ ultimate analysis predictor")
st.markdown("Convert a **proximate analysis** into an **ultimate analysis**, or the other way round, with neural "
            "networks trained on 6,463 US samples (USGS COALQUAL) and 91 Indian samples (USGS–Geological Survey of "
            "India), lignite to anthracite, and tested on 1,239 samples from 60 other countries.")

tab_a, tab_b, tab_batch, tab_about = st.tabs(
    ["Proximate → Ultimate", "Ultimate → Proximate", "Many samples (CSV)", "Model & accuracy"])

# ------------------------------------------------------------------ proximate -> ultimate
with tab_a:
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("1 · Proximate analysis")
        ex = st.selectbox("Load an example (optional)", ["—"] + list(EXAMPLES), key="ex_a")
        if ex != "—" and st.session_state.get("_last_ex") != ex:
            e = EXAMPLES[ex]
            st.session_state["basis"] = next(k for k, v in BASES.items() if v == e["basis"])
            for k in ("M", "VM", "Ash"):
                st.session_state[k] = float(e[k])
            st.session_state["_last_ex"] = ex
        st.session_state.setdefault("basis", list(BASES)[0])
        for k, v in dict(M=6.0, VM=24.0, Ash=35.0).items():
            st.session_state.setdefault(k, v)

        basis_label = st.radio("Basis of your values", list(BASES), key="basis",
                               help="Indian labs usually report proximate analysis on an air-dried basis (IS 1350 "
                                    "Part 1); US data are usually as-received. The ultimate analysis is returned on "
                                    "the same basis, and also on dry and dry-ash-free bases.")
        basis = BASES[basis_label]
        c1, c2, c3 = st.columns(3)
        if basis == "dry":
            M = 0.0
            c1.number_input("Moisture (%)", value=0.0, disabled=True, help="Zero on a dry basis.")
        else:
            M = c1.number_input("Moisture, M (%)", 0.0, 70.0, step=0.1, format="%.2f", key="M")
        VM = c2.number_input("Volatile matter, VM (%)", 0.0, 80.0, step=0.1, format="%.2f", key="VM")
        Ash = c3.number_input("Ash (%)", 0.0, 85.0, step=0.1, format="%.2f", key="Ash")
        FC = 100.0 - M - VM - Ash
        fc_in = st.number_input("Fixed carbon, FC (%) — optional check", value=None, step=0.1, format="%.2f",
                                placeholder=f"by difference: {FC:.2f}",
                                help="Leave empty to use FC = 100 − M − VM − Ash. If you enter your lab value, "
                                     "it is only used to check that the four numbers add up to 100.")
        moist_dry = None
        if basis == "dry":
            moist_dry = st.number_input("Moisture of the original sample (%) — optional, improves accuracy",
                                        value=None, min_value=0.0, max_value=70.0, step=0.1, format="%.2f")
        S_in = st.number_input("Measured total sulphur on the same basis (%) — optional", value=None,
                               min_value=0.0, max_value=20.0, step=0.01, format="%.2f",
                               help="Sulphur cannot be predicted reliably from a proximate analysis. If you have "
                                    "a measured value (IS 1350 Part 3 / ASTM D4239), enter it: it is used as-is "
                                    "and oxygen is recalculated.")

    with right:
        st.subheader("2 · Predicted ultimate analysis")
        errors = []
        if FC < 0:
            errors.append("Moisture + volatile matter + ash is more than 100 %. Please check the values.")
        if VM <= 0:
            errors.append("Volatile matter must be above zero.")
        if fc_in is not None and abs(M + VM + Ash + fc_in - 100) > 1.0:
            errors.append(f"M + VM + Ash + FC = {M + VM + Ash + fc_in:.2f} %, but it should be 100 %. "
                          "Check that all four values are on the same basis.")
        for e in errors:
            st.error(e)
        if not errors:
            try:
                r = model.predict(M, VM, Ash, S=S_in, basis=basis, moisture_for_dry=moist_dry)
            except ValueError as e:
                st.error(str(e))
                r = None
            if r is not None:
                for w in r.warnings:
                    st.warning(w)
                err = meta["accuracy"]["india_mae"]
                tbl = pd.DataFrame({
                    "Element": [NAMES[k] for k in ELEMENTS],
                    f"{basis.capitalize()} basis (%)": [r.basis_values[k] for k in ELEMENTS],
                    "Dry basis (%)": [r.dry[k] for k in ELEMENTS],
                    "Dry, ash-free (%)": [r.daf[k] for k in ELEMENTS],
                    "Typical error ± (pp)": [0.0 if (k == "S" and S_in is not None) else err[k] for k in ELEMENTS],
                })
                st.dataframe(tbl.style.format(precision=2), hide_index=True, use_container_width=True)
                cols = st.columns(5)
                for c, k in zip(cols, ELEMENTS):
                    c.metric(f"{k} (%)", f"{r.basis_values[k]:.2f}")
                total = sum(r.basis_values.values()) + M + Ash
                st.caption(f"Closure check: C + H + N + S + O + moisture + ash = {total:.2f} % · "
                           f"FC (by difference) = {r.fixed_carbon:.2f} % · organic matter = {r.organic:.2f} %")
                if S_in is None:
                    st.info("Sulphur here is only a rough estimate (it depends on pyrite and the depositional "
                            "environment, which a proximate analysis does not show). If your lab measured total "
                            "sulphur, enter it on the left.")
                st.caption("Typical error = mean absolute error on Indian coal mines that the model never saw "
                           "in training (percentage points, i.e. plain difference in wt %).")

# ------------------------------------------------------------------ ultimate -> proximate
with tab_b:
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("1 · Ultimate analysis")
        exu = st.selectbox("Load an example (optional)", ["—"] + list(EXAMPLES_U), key="ex_b")
        if exu != "—" and st.session_state.get("_last_exu") != exu:
            e = EXAMPLES_U[exu]
            st.session_state["ubasis"] = next(k for k, v in BASES_U.items() if v == e["basis"])
            for k in ELEMENTS:
                st.session_state["u" + k] = float(e[k])
            st.session_state["uM"] = e["M"]
            st.session_state["uA"] = None
            st.session_state["_last_exu"] = exu
        st.session_state.setdefault("ubasis", list(BASES_U)[0])
        for k, v in dict(C=47.77, H=3.04, N=1.00, O=6.51, S=0.68).items():
            st.session_state.setdefault("u" + k, v)

        ub_label = st.radio("Basis of your values", list(BASES_U), key="ubasis",
                            help="Ultimate analysis reported excluding the hydrogen and oxygen of moisture, oxygen by "
                                 "difference (the usual convention). On an as-received or air-dried basis "
                                 "C + H + N + S + O + moisture + ash = 100.")
        ubasis = BASES_U[ub_label]
        cols = st.columns(5)
        U = {k: c.number_input(f"{k} (%)", 0.0, 100.0, step=0.01, format="%.2f", key="u" + k)
             for c, k in zip(cols, ELEMENTS)}
        tot = sum(U.values())
        Mk = Ak = None
        if ubasis in ("as-received", "air-dried"):
            st.metric("Moisture + ash (100 − C − H − N − S − O), exact", f"{100 - tot:.2f} %")
            c1, c2 = st.columns(2)
            Mk = c1.number_input("Measured moisture (%) — optional, strongly recommended", value=None,
                                 min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="uM",
                                 help="Without it, the split of moisture + ash into moisture and ash is estimated "
                                      "from coal rank and is the least certain output. With it, ash is exact.")
            Ak = c2.number_input("…or measured ash (%) — optional", value=None, min_value=0.0, max_value=100.0,
                                 step=0.1, format="%.2f", key="uA", disabled=Mk is not None)
            if Mk is not None:
                Ak = None
        elif ubasis == "daf":
            st.caption("On a dry, ash-free basis the moisture and ash are not part of the analysis. Give the "
                       "as-received moisture and ash to get a full proximate analysis; otherwise only VM and FC "
                       "on a dry, ash-free basis are returned.")
            c1, c2 = st.columns(2)
            Mk = c1.number_input("As-received moisture (%) — optional", value=None, min_value=0.0,
                                 max_value=90.0, step=0.1, format="%.2f", key="uMd")
            Ak = c2.number_input("As-received ash (%) — optional", value=None, min_value=0.0,
                                 max_value=90.0, step=0.1, format="%.2f", key="uAd")
            if (Mk is None) != (Ak is None):
                st.info("Enter both moisture and ash to convert to the as-received basis.")
                Mk = Ak = None

    with right:
        st.subheader("2 · Predicted proximate analysis")
        try:
            q = model.predict_proximate(U["C"], U["H"], U["N"], U["O"], U["S"], basis=ubasis, M=Mk, Ash=Ak)
        except ValueError as e:
            st.error(str(e))
            q = None
        if q is not None:
            for w in q.warnings:
                st.warning(w)
            acc = meta["reverse"]["accuracy"]["india"]
            given = q.moisture_source != "predicted"

            def err_of(k):
                if k in ("M", "Ash"):
                    return 0.0 if given else acc[k]
                return acc[k]

            if q.basis_values["M"] is None:
                st.markdown("Only the dry, ash-free values can be given without moisture and ash:")
                tbl = pd.DataFrame({"Parameter": [NAMES["VM"], NAMES["FC"]],
                                    "Dry, ash-free (%)": [q.daf["VM"], q.daf["FC"]],
                                    "Typical error ± (pp)": [acc["VM"], acc["FC"]]})
                st.dataframe(tbl.style.format(precision=2), hide_index=True, use_container_width=True)
            else:
                shown_basis = "as-received" if ubasis == "daf" else ubasis
                tbl = pd.DataFrame({
                    "Parameter": [NAMES[k] for k in PROXIMATE],
                    f"{shown_basis.capitalize()} basis (%)": [q.basis_values[k] for k in PROXIMATE],
                    "Dry basis (%)": [q.dry[k] for k in PROXIMATE],
                    "Dry, ash-free (%)": [np.nan, q.daf["VM"], np.nan, q.daf["FC"]],
                    "Typical error ± (pp)": [err_of(k) for k in PROXIMATE],
                })
                st.dataframe(tbl.style.format(precision=2, na_rep="—"), hide_index=True, use_container_width=True)
                mcols = st.columns(4)
                for c, k in zip(mcols, PROXIMATE):
                    c.metric(f"{k} (%)", f"{q.basis_values[k]:.2f}")
                s = sum(q.basis_values.values())
                st.caption(f"Check: M + VM + Ash + FC = {s:.2f} % · moisture: {q.moisture_source} · "
                           f"organic matter = {q.organic:.2f} %")
                if q.moisture_source == "predicted":
                    st.info("Moisture + ash is exact, but how it splits into moisture and ash was estimated from "
                            "coal rank (typical error ± "
                            f"{acc['M']:.1f} pp on Indian coal). Enter the measured moisture to make both exact.")
            st.caption("Typical error = mean absolute error on Indian coal mines the model never saw in training, "
                       "percentage points. 0.00 = exact by mass balance.")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.subheader("Predict many samples at once")
    direction = st.radio("Direction", ["Proximate → Ultimate", "Ultimate → Proximate"], horizontal=True, key="bdir")
    forward = direction.startswith("Prox")
    if forward:
        st.markdown("CSV columns: **M, VM, Ash** (optional **S** = measured sulphur), all on the basis chosen below.")
        b_label = st.radio("Basis of the file", list(BASES), key="bbasis", horizontal=True)
        tmpl = pd.DataFrame([dict(sample="Indian ADB example", M=6.0, VM=24.0, Ash=35.0, S=np.nan),
                             dict(sample="US ARB example", M=2.5, VM=33.0, Ash=9.0, S=1.2)])
    else:
        st.markdown("CSV columns: **C, H, N, O, S** on an as-received or air-dried basis (optional **M** = measured "
                    "moisture, which makes moisture and ash exact).")
        tmpl = pd.DataFrame([dict(sample="Indian ARB example", C=47.77, H=3.04, N=1.00, O=6.51, S=0.68, M=np.nan),
                             dict(sample="with moisture", C=63.35, H=4.00, N=1.40, O=8.09, S=0.57, M=9.71)])
    st.download_button("Download template CSV", tmpl.to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"], key=f"up_{forward}")
    if up is not None:
        d = pd.read_csv(up)
        d.columns = [c.strip() for c in d.columns]
        num = lambda c: pd.to_numeric(d[c], errors="coerce").values  # noqa: E731
        out = d.copy()
        if forward:
            bb = BASES[b_label]
            need = ["VM", "Ash"] + ([] if bb == "dry" else ["M"])
            miss = [c for c in need if c not in d.columns]
            if miss:
                st.error("Missing columns: " + ", ".join(miss))
            else:
                Mv = np.zeros(len(d)) if bb == "dry" else num("M")
                VMv, Av = num("VM"), num("Ash")
                Sv = num("S") if "S" in d.columns else None
                ok = ~np.isnan(Mv) & ~np.isnan(VMv) & ~np.isnan(Av) & (100 - Mv - Av - VMv >= 0) & (VMv > 0) \
                    & (Mv >= 0) & (Av >= 0)
                for k in ELEMENTS:
                    out["pred_" + k] = np.nan
                if ok.any():
                    P = model.predict_arrays(Mv[ok], VMv[ok], Av[ok], None if Sv is None else Sv[ok],
                                             moisture_known=(bb != "dry"))
                    for j, k in enumerate(ELEMENTS):
                        out.loc[ok, "pred_" + k] = P[:, j].round(2)
                if (~ok).any():
                    st.warning(f"{int((~ok).sum())} row(s) skipped: blank, negative, or M + VM + Ash > 100.")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download results", out.to_csv(index=False), "ultimate_predictions.csv", "text/csv")
        else:
            miss = [c for c in ELEMENTS if c not in d.columns]
            if miss:
                st.error("Missing columns: " + ", ".join(miss))
            else:
                U = {k: num(k) for k in ELEMENTS}
                Mv = num("M") if "M" in d.columns else None
                tot = sum(U.values())
                ok = np.all([~np.isnan(U[k]) & (U[k] >= 0) for k in ELEMENTS], axis=0) & (U["C"] > 0) & (U["H"] > 0) \
                    & (tot <= 100) & (tot >= 15)
                for k in PROXIMATE:
                    out["pred_" + k] = np.nan
                if ok.any():
                    P = model.predict_proximate_arrays(*(U[k][ok] for k in ELEMENTS), M=None if Mv is None else Mv[ok])
                    for j, k in enumerate(PROXIMATE):
                        out.loc[ok, "pred_" + k] = P[:, j].round(2)
                if (~ok).any():
                    st.warning(f"{int((~ok).sum())} row(s) skipped: blank or negative values, C or H zero, or "
                               "C + H + N + O + S outside 15–100 %.")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download results", out.to_csv(index=False), "proximate_predictions.csv", "text/csv")

# ------------------------------------------------------------------ about
with tab_about:
    acc, racc = meta["accuracy"], meta["reverse"]["accuracy"]
    st.subheader("How the models work")
    st.markdown(
        "**Proximate → ultimate.** Organic matter = 100 − M − Ash; volatile matter on a dry, ash-free basis "
        "VM_daf = 100·VM/(100 − M − Ash). The composition of the organic matter depends mainly on coal rank, which "
        "makes US, world and Indian coals comparable and the method independent of the reporting basis. An "
        f"ensemble of {meta['n_mlps']} neural networks (hidden layers {' × '.join(map(str, meta['hidden']))}), "
        "blended 70 : 30 with a quadratic regression, reads VM_daf, Ash/organic and M/organic and predicts C, H, N "
        "and S (daf); oxygen is by difference, so C + H + N + S + O + M + Ash = 100 % exactly.\n\n"
        "**Ultimate → proximate.** C + H + N + S + O gives the organic matter, so moisture + ash is known exactly. "
        "One network ensemble predicts VM_daf from the daf elemental composition (then FC = organic − VM); a second "
        "estimates how moisture + ash splits into moisture and ash — this split is exact when the measured moisture "
        "is supplied.")
    st.subheader("Accuracy on coal the models never saw")
    st.markdown("Mean absolute error in percentage points. Indian: each mine or block held out in turn "
                "(leave-one-site-out). US: whole sample groups held out (grouped 5-fold). World: 60 countries "
                "never used in training.")
    st.markdown("**Proximate → ultimate**")
    st.dataframe(pd.DataFrame(acc["table"]).style.format(precision=2), hide_index=True, use_container_width=True)
    st.markdown("**Ultimate → proximate** (as-received; M and Ash become exact when moisture is given)")
    st.dataframe(pd.DataFrame([racc["india"], racc["us"]]).style.format(precision=2), hide_index=True,
                 use_container_width=True)
    st.subheader("Training data")
    st.dataframe(pd.DataFrame(meta["data"]), hide_index=True, use_container_width=True)
    st.markdown("**Limitations.** Sulphur is not predictable from proximate analysis (enter it if measured). "
                "Without a measured moisture, the moisture/ash split in the reverse direction is the least certain "
                "output. Hydrogen and oxygen exclude the moisture in the sample (the usual reporting convention). "
                "Indian training data cover the Sohagpur coalfield and Rajasthan lignite; for other coalfields "
                "errors may be somewhat larger.")
    st.caption("Data: Palmer et al. (2015) USGS COALQUAL v3; Tewalt et al. (2010) USGS OFR 2010-1196 (WoCQI); "
               "Trippi & Tewalt (2011) USGS OFR 2011-1296, with the Geological Survey of India.")
