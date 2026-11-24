"""Coal ultimate-analysis predictor (web app).

Enter a proximate analysis (moisture, volatile matter, ash, fixed carbon) and get the
ultimate analysis (C, H, N, S, O). Works for US (ASTM, as-received) and Indian
(IS 1350, air-dried) reporting conventions.

Run locally:   streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import streamlit as st

from coal_model import ELEMENTS, CoalModel

NAMES = {"C": "Carbon", "H": "Hydrogen", "N": "Nitrogen", "O": "Oxygen (by difference)", "S": "Sulphur"}
BASES = {"As-received (ARB) — US / ASTM D3172 & D3176 style": "as-received",
         "Air-dried (ADB) — Indian IS 1350 style": "air-dried",
         "Dry (DB)": "dry"}
EXAMPLES = {
    "Indian non-coking coal, high ash (Sohagpur-type, ADB)": dict(basis="air-dried", M=6.0, VM=24.0, Ash=35.0),
    "Indian power-plant coal, G11-type (ADB)": dict(basis="air-dried", M=7.0, VM=22.0, Ash=40.0),
    "Indian coking coal, washed (Jharia-type, ADB)": dict(basis="air-dried", M=1.2, VM=24.0, Ash=17.0),
    "Indian lignite (Rajasthan / Neyveli-type, ARB)": dict(basis="as-received", M=35.0, VM=28.0, Ash=10.0),
    "US Appalachian bituminous (ARB)": dict(basis="as-received", M=2.5, VM=33.0, Ash=9.0),
    "US Powder River Basin sub-bituminous (ARB)": dict(basis="as-received", M=27.0, VM=31.0, Ash=5.5),
    "US Pennsylvania anthracite (ARB)": dict(basis="as-received", M=4.0, VM=5.0, Ash=11.0),
}

st.set_page_config(page_title="Coal Ultimate Analysis Predictor", page_icon="⛏️", layout="wide")


@st.cache_resource
def load_model():
    return CoalModel()


model = load_model()
meta = model.meta

st.title("⛏️ Coal ultimate-analysis predictor")
st.markdown("Enter the **proximate analysis** of a coal and a neural network predicts its **ultimate analysis**. "
            "Trained on 6,463 US samples (USGS COALQUAL) and 91 Indian samples (USGS–Geological Survey of India), "
            "lignite to anthracite, and tested on 1,239 samples from 60 other countries.")

tab_one, tab_batch, tab_about = st.tabs(["Single sample", "Many samples (CSV)", "Model & accuracy"])

# ------------------------------------------------------------------ single sample
with tab_one:
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("1 · Proximate analysis")
        ex = st.selectbox("Load an example (optional)", ["—"] + list(EXAMPLES))
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
                            "sulphur, enter it above.")
                st.caption("Typical error = mean absolute error on Indian coal mines that the model never saw "
                           "in training (percentage points, i.e. plain difference in wt %).")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.subheader("Predict many samples at once")
    st.markdown("Upload a CSV with columns **M, VM, Ash** (optional: **S** = measured sulphur). "
                "All values on the same basis, which you choose below.")
    b_label = st.radio("Basis of the file", list(BASES)[:2] + ["Dry (DB)"], key="bbasis", horizontal=True)
    tmpl = pd.DataFrame([dict(sample="Indian ADB example", M=6.0, VM=24.0, Ash=35.0, S=np.nan),
                         dict(sample="US ARB example", M=2.5, VM=33.0, Ash=9.0, S=1.2)])
    st.download_button("Download template CSV", tmpl.to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        d = pd.read_csv(up)
        d.columns = [c.strip() for c in d.columns]
        bb = BASES[b_label]
        if bb == "dry" and "M" not in d.columns:
            d["M"] = 0.0
        miss = [c for c in ("M", "VM", "Ash") if c not in d.columns]
        if miss:
            st.error("Missing columns: " + ", ".join(miss))
        else:
            Mv = np.zeros(len(d)) if bb == "dry" else pd.to_numeric(d.M, errors="coerce").values
            VMv = pd.to_numeric(d.VM, errors="coerce").values
            Av = pd.to_numeric(d.Ash, errors="coerce").values
            Sv = pd.to_numeric(d.S, errors="coerce").values if "S" in d.columns else None
            ok = ~np.isnan(Mv) & ~np.isnan(VMv) & ~np.isnan(Av) & (100 - Mv - Av - VMv >= 0) & (VMv > 0) \
                & (Mv >= 0) & (Av >= 0)
            out = d.copy()
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

# ------------------------------------------------------------------ about
with tab_about:
    acc = meta["accuracy"]
    st.subheader("How the model works")
    st.markdown(
        "1. **Remove moisture and ash.** Organic matter = 100 − M − Ash. Volatile matter is expressed on a "
        "dry, ash-free (daf) basis, VM_daf = 100·VM / (100 − M − Ash). Composition of the organic matter depends "
        "mainly on coal rank, so this makes US, world and Indian coals comparable and makes the method work for "
        "any reporting basis.\n"
        "2. **Neural network.** Inputs: VM_daf, Ash/organic and M/organic. An ensemble of "
        f"{meta['n_mlps']} neural networks (hidden layers {' × '.join(map(str, meta['hidden']))}), blended "
        "70 : 30 with a quadratic regression, predicts C, H, N and S on a daf basis.\n"
        "3. **Oxygen by difference** and conversion back to your basis, so that "
        "C + H + N + S + O + M + Ash = 100 % exactly.")
    st.subheader("Accuracy on coal the model never saw")
    st.markdown("Mean absolute error in percentage points. Indian: each mine/coalfield block held out in turn "
                "(leave-one-site-out). US: whole sample groups held out (grouped 5-fold).")
    st.dataframe(pd.DataFrame(acc["table"]).style.format(precision=2), hide_index=True, use_container_width=True)
    st.subheader("Training data")
    st.dataframe(pd.DataFrame(meta["data"]), hide_index=True, use_container_width=True)
    st.markdown("**Limitations.** Sulphur is not predictable from proximate analysis (enter it if measured). "
                "Hydrogen and oxygen exclude the moisture in the sample (standard ‘moisture-free’ reporting of "
                "ultimate analysis on a given basis). Indian training data cover the Sohagpur coalfield and "
                "Rajasthan lignite; for other coalfields errors may be somewhat larger.")
    st.caption("Data: Palmer et al. (2015) USGS COALQUAL v3; Tewalt et al. (2010) USGS OFR 2010-1196 (WoCQI); "
               "Trippi & Tewalt (2011) USGS OFR 2011-1296, with the Geological Survey of India.")
