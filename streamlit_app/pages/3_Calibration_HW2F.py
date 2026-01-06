from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_app.ui.io import load_curve_and_swaption_from_upload
from streamlit_app.ui.capture import capture_stdout
from streamlit_app.ui.plotting import fig_curve, fig_prices_by_tenor, fig_vols_by_tenor

from ir.market.loaders_excel import load_curve_xlsx, load_swaption_template_xlsx
from ir.pricers.hw2f_pricer import HullWhite2FPricer
from ir.calibration.hw2f_profile import HullWhite2FProfileCalibrator
from ir.calibration.vol import black_normal_vol
from streamlit_app.ui.db import curve_to_dict


# -----------------------------
# FICHIER PAR DÉFAUT (dans le repo)
# -----------------------------
# Si l’utilisateur n’upload rien, on charge ce template local (si présent).
ROOT = Path(__file__).resolve().parents[2]  # racine du projet
DEFAULT_REL = Path("Calibration_Templates") / "SWPN_Calibration_Template_30092025_USD.xlsx"
DEFAULT_XLSX = ROOT / DEFAULT_REL


def ensure_expiry_tenor(df: pd.DataFrame, dates_col="Payment_Dates"):
    """
    Ajoute (si absentes) les colonnes :
      - Expiry = T0 (début de la schedule de paiement)
      - Tenor  = Tn - T0 (maturité du swap sous-jacent)

    Utile pour trier/plotter par (Expiry, Tenor) ensuite.
    """
    if "Expiry" not in df.columns:
        df["Expiry"] = df[dates_col].apply(lambda L: float(L[0]))
    if "Tenor" not in df.columns:
        df["Tenor"] = df[dates_col].apply(lambda L: float(L[-1]) - float(L[0]))


def par_rate(curve, Tau):
    """
    Calcule (A0, S0) à t=0 pour un échéancier fixe Tau=[T0, T1, ..., Tn].

    - A0 : annuité, A0 = ∑_{i=1..n} Delta_i * P(0,Ti)
    - S0 : taux swap par, S0 = (P(0,T0) - P(0,Tn)) / A0

    Retour :
      (A0, S0) en unités "rate" (S0 ~ 0.03)
    """
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
    """
    Calcule des vols implicites normales (Bachelier) à partir des prix (forward premium).

    Hypothèse/convention :
      - Les prix comparés sont des "forward premiums" (PV/DF(T0)).
      - Pour l'inversion Bachelier, on utilise une "annuité forward" :
            annuity_fwd = A0 / DF(T0)

    Ajoute deux colonnes :
      - Market_Vol (Bps)
      - Model_Vol (Bps)
    """
    mkt_vol, mdl_vol = [], []

    for _, row in df.iterrows():
        Tau = row[dates_col]
        T0 = float(Tau[0])

        # DF0 sert à passer de l’annuité "PV" à une annuité "forward"
        DF0 = float(curve.discount(T0))

        # (A0,S0) pour calculer forward swap rate et annuity_fwd
        A0, S0 = par_rate(curve, Tau)
        annuity_fwd = A0 / (DF0 + 1e-18)

        # Strike du template en % ; forward en % aussi (pour black_normal_vol)
        strike_pct = float(row[strike_col])
        forward_pct = 100.0 * float(S0)

        # Notional : pris dans la ligne si existant, sinon 1.0
        notional = float(row.get("Notional", 1.0))

        # Prix forward premium (marché vs modèle)
        p_mkt = float(row[price_col])
        p_mdl = float(row[model_col])

        # Inversion Bachelier -> vols normales implicites en bps
        mkt_vol.append(black_normal_vol(p_mkt, forward_pct, strike_pct, T0, notional, annuity_fwd))
        mdl_vol.append(black_normal_vol(p_mdl, forward_pct, strike_pct, T0, notional, annuity_fwd))

    df["Market_Vol (Bps)"] = mkt_vol
    df["Model_Vol (Bps)"] = mdl_vol


# =============================
# UI STREAMLIT : PAGE HW2F
# =============================
st.markdown("# Calibration — Hull–White 2F (G2++)")
st.caption("Profile calibration : outer (a,b,rho) + inner (sigma,eta) + live progress.")

# Colonne gauche = inputs ; colonne droite = visualisation/résultats
colL, colR = st.columns([1.0, 1.2], gap="large")

with colL:
    # Upload du template excel
    uploaded = st.file_uploader("Upload SWPN calibration template (.xlsx)", type=["xlsx"], key="hw2f_upload")
    st.caption(
        f"Si aucun fichier n’est uploadé, l’app utilise par défaut : `{DEFAULT_REL.as_posix()}`"
        if DEFAULT_XLSX.exists()
        else "Aucun fichier par défaut trouvé dans `Calibration_Templates/`."
    )

    curve_sheet = st.text_input("Curve sheet", value="Curve", key="hw2f_curve_sheet")
    template_sheet = st.text_input("Template sheet", value="Template", key="hw2f_template_sheet")
    smooth = st.number_input("Curve smoothing", value=1e-7, format="%.1e", key="hw2f_smooth")

    st.divider()
    st.subheader("Outer grid (coarse)")

    # Grille grossière de recherche sur (a,b,rho) (profile calibration)
    grid_a = st.text_input("grid_a", value="0.01,0.02,0.05,0.10,0.20")
    grid_b = st.text_input("grid_b", value="0.001,0.003,0.01,0.02,0.05")
    grid_rho = st.text_input("grid_rho", value="-0.8,-0.5,-0.2,0.0,0.2")

    st.subheader("Inner init")
    init_sigma = st.number_input("init sigma", value=0.01, format="%.6f", key="hw2f_init_sigma")
    init_eta = st.number_input("init eta", value=0.008, format="%.6f", key="hw2f_init_eta")

    verbose_inner = st.checkbox("Verbose inner prints", value=False)
    top_k = st.number_input("top_k candidates printed", value=3, step=1, min_value=1)

    do_calibrate = st.button("🚀 Run profile calibration (HW2F)", type="primary", use_container_width=True)

with colR:
    # -----------------------------
    # CHARGEMENT : upload OU fichier par défaut
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

    # Affichage de la courbe
    st.subheader("Courbe de Marché")
    st.pyplot(fig_curve(curve, title_prefix="Marché"), clear_figure=True)

    # Aperçu du template
    st.subheader("Aperçu du template de calibration")
    st.dataframe(swpn.df.head(15), use_container_width=True, height=260)

    # Placeholders pour retour visuel (status + barre + tableau)
    status_ph = st.empty()
    bar_ph = st.progress(0)
    progress_ph = st.empty()

    if do_calibrate:
        st.session_state["hw2f_progress_rows"] = []

        def _parse_list(s: str):
            """
            Parse une chaîne "0.01,0.02,0.05" -> [0.01, 0.02, 0.05]
            """
            return [float(x.strip()) for x in s.split(",") if x.strip()]

        ga = _parse_list(grid_a)
        gb = _parse_list(grid_b)
        gr = _parse_list(grid_rho)

        def progress_cb(d: dict):
            """
            Callback appelée par le calibrator HW2F pendant la boucle outer.

            - Met à jour une barre de progression basée sur outer_idx/outer_total
            - Affiche un statut texte (candidat courant + best RMSRE)
            - Affiche un tableau "dernier historique" (tail 25)
            """
            rows = st.session_state.get("hw2f_progress_rows", [])
            rows.append(d)
            st.session_state["hw2f_progress_rows"] = rows

            outer_idx = int(d.get("outer_idx", 0) or 0)
            outer_total = int(d.get("outer_total", 0) or 0)
            pct = 0
            if outer_total > 0:
                pct = int(round(100 * outer_idx / max(outer_total, 1)))
            bar_ph.progress(min(max(pct, 0), 100))

            # Informations candidates
            stage = d.get("stage", "")
            a = d.get("a", None)
            b = d.get("b", None)
            rho = d.get("rho", None)
            cand_rmsre = d.get("cand_rmsre", None)
            best_rmsre = d.get("best_rmsre", None)
            improved = d.get("improved", False)

            # Message de statut selon étape
            if stage == "outer_start":
                status_ph.markdown(
                    f"**HW2F calibration en cours** — candidat **{outer_idx}/{outer_total}** ({pct}%)  \n"
                    f"`a={a:.6f}` | `b={b:.6f}` | `rho={rho:.3f}`  \n"
                    f"Best RMSRE so far: `{(best_rmsre if best_rmsre is not None else float('nan')):.2e}`"
                )
            elif stage == "outer_done":
                status_ph.markdown(
                    f"**HW2F calibration en cours** — candidat **{outer_idx}/{outer_total}** ({pct}%)  \n"
                    f"`a={a:.6f}` | `b={b:.6f}` | `rho={rho:.3f}`  \n"
                    f"Cand RMSRE: `{(cand_rmsre if cand_rmsre is not None else float('nan')):.2e}`  |  "
                    f"Best: `{(best_rmsre if best_rmsre is not None else float('nan')):.2e}`"
                    + ("  ✅ **NEW BEST**" if improved else "")
                )
            else:
                status_ph.markdown(f"**HW2F calibration** — {outer_idx}/{outer_total} ({pct}%)")

            # Tableau des dernières lignes de progression
            dfp = pd.DataFrame(rows)
            cols = [
                c
                for c in [
                    "stage",
                    "outer_idx",
                    "outer_total",
                    "a",
                    "b",
                    "rho",
                    "cand_rmsre",
                    "best_rmsre",
                    "improved",
                ]
                if c in dfp.columns
            ]
            progress_ph.dataframe(
                dfp[cols].tail(25) if cols else dfp.tail(25),
                use_container_width=True,
                hide_index=True,
                height=260,
            )

        # Lancement calibration (avec spinner) et capture des prints
        with st.spinner("Calibration HW2F (profile) en cours..."):
            # Pricer 2F : ici analytique (approx swap-rate) / ZC options
            pricer_2f = HullWhite2FPricer(curve)

            # QuoteSet -> dict marché attendu par le calibrator
            mkt_dict = swpn.to_market_dict()

            # Calibrator profile (outer grid + inner optimization)
            cal = HullWhite2FProfileCalibrator(
                pricer_2f,
                mkt_dict,
                use_forward_premium=True,
                progress_cb=progress_cb,
            )

            # Exécute et récupère (résultat, logs)
            res, logs = capture_stdout(
                cal.calibrate_profile,
                grid_a=ga,
                grid_b=gb,
                grid_rho=gr,
                init_sigma=init_sigma,
                init_eta=init_eta,
                verbose_inner=verbose_inner,
                top_k=int(top_k),
            )

            st.session_state["hw2f_pricer"] = pricer_2f
            st.session_state["hw2f_logs"] = logs
            st.session_state["hw2f_profile_res"] = res

        # Message final (si possible avec best rmsre)
        try:
            best = (res or {}).get("best", {}) if isinstance(res, dict) else {}
            bar_ph.progress(100)
            status_ph.success(
                f"Calibration terminée ✅  |  Best RMSRE={(best.get('rmsre', float('nan'))):.2e}",
                icon="✅",
            )
        except Exception:
            bar_ph.progress(100)
            status_ph.success("Calibration terminée ✅", icon="✅")

    # -----------------------------
    # SECTION "POST-RUN" : affichage résultats si un run existe en session
    # -----------------------------
    if "hw2f_pricer" in st.session_state:
        pricer_2f = st.session_state["hw2f_pricer"]
        res = st.session_state.get("hw2f_profile_res", {})
        best = (res or {}).get("best", {})

        st.subheader("Calibration logs")
        with st.expander("Voir logs (print calibrator)", expanded=False):
            st.code(st.session_state.get("hw2f_logs", ""), language="text")

        # Historique candidats outer 
        rows = st.session_state.get("hw2f_progress_rows", [])
        if rows:
            with st.expander("Progress (candidats outer)", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=320)

        # Paramètres best (a,b,rho,sigma,eta,rmsre) renvoyés par le calibrator
        st.subheader("Best parameters")
        st.json(best)

        # Comparaison mkt vs modèle : pricing 2F (swaption approx) sur tout le template
        st.subheader("Market vs Model")
        df_2f = swpn.with_model_prices_2f(pricer_2f, forward_premium=True)
        ensure_expiry_tenor(df_2f)

        st.dataframe(
            df_2f[["Expiry", "Tenor", "Strike", "Price", "Model_Price", "Rel_Error"]].head(30),
            use_container_width=True,
            height=260,
        )

        # Plots prix et vols vs expiries (par tenor)
        st.pyplot(
            fig_prices_by_tenor(df_2f, mkt_col="Price", model_col="Model_Price", ylabel="Forward Premium"),
            clear_figure=True,
        )

        with st.spinner("Implied vols (Bachelier) ..."):
            dfv = df_2f.copy()
            add_implied_normal_vols_forward_premium(dfv, curve)

        st.pyplot(
            fig_vols_by_tenor(dfv, mkt_col="Market_Vol (Bps)", model_col="Model_Vol (Bps)"),
            clear_figure=True,
        )

        params = dict(pricer_2f.model.parameters)
        rmsre = best.get("rmsre", None)

        curve_snapshot = curve_to_dict(curve)
        st.session_state["last_curve_snapshot"] = curve_snapshot

        st.session_state["last_run"] = {
            "model": "HW2F",
            "source_file": st.session_state.get("last_source_file"),
            "params": params,
            "rmsre": rmsre,
            "artifacts": {},
            "curve_snapshot": curve_snapshot,
        }

        st.info("Tu peux maintenant aller sur **PFE Swap** (page 4).", icon="➡️")
