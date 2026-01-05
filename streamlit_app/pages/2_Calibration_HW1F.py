from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.ui.io import load_curve_and_swaption_from_upload
from streamlit_app.ui.capture import capture_stdout
from streamlit_app.ui.plotting import fig_curve, fig_prices_by_tenor, fig_vols_by_tenor

from ir.market.loaders_excel import load_curve_xlsx, load_swaption_template_xlsx
from ir.pricers.hw1f_pricer import HullWhitePricer
from ir.calibration.hw1f_calibration import HullWhiteCalibrator
from ir.calibration.vol import black_normal_vol
from streamlit_app.ui.db import save_run, curve_to_dict


# -----------------------------
# DEFAULT FILE (repo-local)
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
DEFAULT_REL = Path("Calibration_Templates") / "SWPN_Calibration_Template_30092025_USD.xlsx"
DEFAULT_XLSX = ROOT / DEFAULT_REL


def ensure_expiry_tenor(df: pd.DataFrame, dates_col="Payment_Dates"):
    if "Expiry" not in df.columns:
        df["Expiry"] = df[dates_col].apply(lambda L: float(L[0]))
    if "Tenor" not in df.columns:
        df["Tenor"] = df[dates_col].apply(lambda L: float(L[-1]) - float(L[0]))


def par_rate(curve, Tau):
    Tau = [float(x) for x in Tau]
    T0, Tn = Tau[0], Tau[-1]
    A0 = 0.0
    for i in range(1, len(Tau)):
        Ti = Tau[i]
        d = Tau[i] - Tau[i - 1]
        A0 += d * float(curve.discount(Ti))
    S0 = (float(curve.discount(T0)) - float(curve.discount(Tn))) / (A0 + 1e-18)
    return A0, S0


def add_implied_normal_vols_forward_premium(
    df: pd.DataFrame,
    curve,
    price_col="Price",
    model_col="Model_Price",
    strike_col="Strike",
    dates_col="Payment_Dates",
):
    mkt_vol, mdl_vol = [], []
    for _, row in df.iterrows():
        Tau = row[dates_col]
        T0 = float(Tau[0])
        DF0 = float(curve.discount(T0))
        A0, S0 = par_rate(curve, Tau)
        annuity_fwd = A0 / (DF0 + 1e-18)

        strike_pct = float(row[strike_col])  # %
        forward_pct = 100.0 * float(S0)      # %
        notional = float(row.get("Notional", 1.0))

        p_mkt = float(row[price_col])
        p_mdl = float(row[model_col])

        mkt_vol.append(black_normal_vol(p_mkt, forward_pct, strike_pct, T0, notional, annuity_fwd))
        mdl_vol.append(black_normal_vol(p_mdl, forward_pct, strike_pct, T0, notional, annuity_fwd))

    df["Market_Vol (Bps)"] = mkt_vol
    df["Model_Vol (Bps)"] = mdl_vol


st.markdown("# Calibration — Hull–White 1F")
st.caption("Calibrage sur swaptions (forward premium) + plots + sauvegarde en session.")

colL, colR = st.columns([1.0, 1.2], gap="large")

with colL:
    uploaded = st.file_uploader("Upload SWPN calibration template (.xlsx)", type=["xlsx"])
    st.caption(
        f"Si aucun fichier n’est uploadé, l’app utilise par défaut : `{DEFAULT_REL.as_posix()}`"
        if DEFAULT_XLSX.exists()
        else "Aucun fichier par défaut trouvé dans `Calibration_Templates/`."
    )

    curve_sheet = st.text_input("Curve sheet", value="Curve")
    template_sheet = st.text_input("Template sheet", value="Template")
    smooth = st.number_input("Curve smoothing", value=1e-7, format="%.1e")

    st.divider()
    st.subheader("Calibration settings")
    init_a = st.number_input("init a", value=0.01, format="%.6f")
    init_sigma = st.number_input("init sigma", value=0.01, format="%.6f")
    method = st.selectbox("Optimizer", ["L-BFGS-B", "Nelder-Mead"], index=0)

    do_calibrate = st.button("🚀 Run calibration (HW1F)", type="primary", use_container_width=True)

with colR:
    # -----------------------------
    # LOAD: uploaded OR default file
    # -----------------------------
    if uploaded is not None:
        source_path, curve, swpn = load_curve_and_swaption_from_upload(
            uploaded, curve_sheet=curve_sheet, template_sheet=template_sheet, smooth=smooth
        )
        source_name = uploaded.name
    else:
        if not DEFAULT_XLSX.exists():
            st.info("Upload un fichier .xlsx pour commencer (template par défaut introuvable).", icon="📄")
            st.stop()

        st.info(
            f"Aucun fichier uploadé — utilisation du template par défaut : `{DEFAULT_REL.as_posix()}`",
            icon="📄",
        )
        source_path = str(DEFAULT_XLSX)
        source_name = DEFAULT_XLSX.name
        curve = load_curve_xlsx(source_path, sheet=curve_sheet, smooth=float(smooth))
        swpn = load_swaption_template_xlsx(source_path, sheet=template_sheet)

    st.session_state["last_source_file"] = source_name

    st.subheader("Market curve")
    st.pyplot(fig_curve(curve, title_prefix="Market"), clear_figure=True)

    st.subheader("Template preview")
    st.dataframe(swpn.df.head(15), use_container_width=True, height=260)

    status_ph = st.empty()
    progress_ph = st.empty()

    if do_calibrate:
        st.session_state["hw1f_progress_rows"] = []

        def progress_cb(d: dict):
            rows = st.session_state.get("hw1f_progress_rows", [])
            rows.append(d)
            st.session_state["hw1f_progress_rows"] = rows

            status_ph.markdown(
                f"**Calibration en cours** — itération **{d['iter']}**  \n"
                f"`a={d['a']:.6f}`  |  `sigma={d['sigma']:.6f}`  |  `RMSRE={d['rmsre']:.2e}`"
            )

            dfp = pd.DataFrame(rows)
            progress_ph.dataframe(
                dfp.tail(20),
                use_container_width=True,
                hide_index=True,
                height=240,
            )

        with st.spinner("Calibration HW1F en cours..."):
            pricer_1f = HullWhitePricer(curve, n_paths=20000, seed=2025)
            mkt_dict = swpn.to_market_dict()

            cal = HullWhiteCalibrator(
                pricer_1f,
                mkt_dict,
                calibrate_to="Swaptions",
                progress_cb=progress_cb,
            )

            result_obj, logs = capture_stdout(
                cal.calibrate,
                init_a=init_a,
                init_sigma=init_sigma,
                method=method,
            )

            st.session_state["hw1f_pricer"] = pricer_1f
            st.session_state["hw1f_logs"] = logs
            st.session_state["hw1f_result"] = result_obj

        try:
            final_a = float(pricer_1f.model.parameters.get("a", np.nan))
            final_sigma = float(pricer_1f.model.parameters.get("sigma", np.nan))
            final_rmsre = float(getattr(result_obj, "fun", np.nan))
            status_ph.success(
                f"Calibration terminée ✅  |  a={final_a:.6f}  sigma={final_sigma:.6f}  RMSRE={final_rmsre:.2e}",
                icon="✅",
            )
        except Exception:
            status_ph.success("Calibration terminée ✅", icon="✅")

    if "hw1f_pricer" in st.session_state:
        pricer_1f = st.session_state["hw1f_pricer"]

        st.subheader("Calibration logs")
        with st.expander("Voir logs (print calibrator)", expanded=False):
            st.code(st.session_state.get("hw1f_logs", ""), language="text")

        rows = st.session_state.get("hw1f_progress_rows", [])
        if rows:
            with st.expander("Progress (itérations)", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=260)

        st.subheader("Market vs Model")
        df_1f = swpn.with_model_prices_1f(pricer_1f, forward_premium=True)
        ensure_expiry_tenor(df_1f)

        st.dataframe(
            df_1f[["Expiry", "Tenor", "Strike", "Price", "Model_Price", "Rel_Error"]].head(30),
            use_container_width=True,
            height=260,
        )

        st.pyplot(
            fig_prices_by_tenor(df_1f, mkt_col="Price", model_col="Model_Price", ylabel="Forward Premium"),
            clear_figure=True,
        )

        with st.spinner("Implied vols (Bachelier) ..."):
            dfv = df_1f.copy()
            add_implied_normal_vols_forward_premium(dfv, curve)

        st.pyplot(
            fig_vols_by_tenor(dfv, mkt_col="Market_Vol (Bps)", model_col="Model_Vol (Bps)"),
            clear_figure=True,
        )

        params = dict(pricer_1f.model.parameters)
        try:
            rmsre = float(getattr(st.session_state.get("hw1f_result", None), "fun", np.nan))
        except Exception:
            rmsre = None

        curve_snapshot = curve_to_dict(curve)
        st.session_state["last_curve_snapshot"] = curve_snapshot

        st.session_state["last_run"] = {
            "model": "HW1F",
            "source_file": st.session_state.get("last_source_file"),
            "params": params,
            "rmsre": rmsre,
            "artifacts": {},
            "curve_snapshot": curve_snapshot,
        }

        st.info("Tu peux maintenant aller sur **PFE Swap** (page 4).", icon="➡️")
