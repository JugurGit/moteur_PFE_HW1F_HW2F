# streamlit_app/pages/4_PFE_Swap.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.ui.plotting import fig_pfe
from streamlit_app.ui.db import list_runs, get_run, format_run_label, curve_from_dict

from ir.risk.pfe_swap import pfe_profile_swap
from ir.risk.hw2f_sim import HW2FCurveSim
from ir.pricers.hw1f_pricer import HullWhitePricer
from ir.pricers.hw2f_pricer import HullWhite2FPricer


# =============================
# PAGE : PFE SWAP
# =============================
st.markdown("# PFE Swap")
st.caption("Calcule PFE/EPE pour un swap vanilla, en réutilisant un run (session ou DB) sans recalibrer.")

# -------------------------
# 1) Sélection de la source du run
# -------------------------
# Objectif : récupérer (modèle, params, courbe) depuis :
#   - Session (last_run) : résultat de la calibration HW1F/HW2F en session
#   - Database : runs sauvegardés en base (via pages calibration)
st.subheader("Run source")

src = st.radio(
    "Choix des paramètres modèle/courbe",
    ["Session (last calibration)", "Database (saved runs)"],
    horizontal=True,
)

selected_run = None
run_model = None
run_params = None
run_curve = None

if src.startswith("Session"):
    last_run = st.session_state.get("last_run", None)
    if last_run is None:
        st.warning(
            "Aucun `last_run` en session. Va d’abord sur Calibration HW1F/HW2F (pages 2/3) ou charge un run DB.",
            icon="⚠️",
        )
    else:
        selected_run = last_run
        run_model = last_run.get("model")
        run_params = last_run.get("params")
        run_curve = None  
        st.info(f"Run session détecté: model={run_model}, source={last_run.get('source_file')}", icon="🧠")

else:
    # Cas DB : on liste les runs sauvegardés et on en charge un
    runs = list_runs(limit=200)
    if not runs:
        st.warning("Aucun run en DB. Sauvegarde un run depuis Calibration HW1F/HW2F.", icon="⚠️")
    else:
        options = {format_run_label(r): r["id"] for r in runs}
        label = st.selectbox("Sélectionne un run", list(options.keys()))
        rid = int(options[label])
        db_run = get_run(rid)
        if db_run is None:
            st.error("Run introuvable en DB (id invalide).")
        else:
            if not db_run.get("curve") or not db_run.get("params"):
                st.error("Run DB incomplet: il manque `curve` ou `params`. Re-sauvegarde un run avec la version patchée.")
            else:
                selected_run = db_run
                run_model = db_run.get("model")
                run_params = db_run.get("params")
                run_curve = curve_from_dict(db_run.get("curve"))  # reconstruction Curve depuis dict JSON
                st.success(f"Run DB chargé ✅  (id={rid}, model={run_model})", icon="✅")

st.divider()

# -------------------------
# 2) Inputs du swap + paramètres Monte Carlo
# -------------------------
# On définit le swap (notional, K, schedule Tau) + la grille de temps pour calculer PFE/EPE
st.subheader("Swap inputs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    q = st.slider("Quantile q", min_value=0.90, max_value=0.995, value=0.95, step=0.005)
    payer = st.checkbox("Payer swap", value=True)

with col2:
    notional = st.number_input("Notional", value=1_000_000.0, step=100_000.0)
    K = st.number_input("Fixed rate K (rate units)", value=0.03, format="%.6f")

with col3:
    grid_n = st.number_input("Grid points", value=21, min_value=5, step=1)
    tau_str = st.text_input("Tau schedule (years)", value="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")

with col4:
    # Paramètres MC génériques (valables 1F et 2F côté simulateur)
    n_paths = st.number_input("MC paths", value=20000, min_value=1000, step=1000)
    seed = st.number_input("Seed", value=2025, step=1)
    n_steps_1f = st.number_input("HW1F steps (Euler)", value=252, min_value=50, step=10)


def _parse_tau(s: str):
    """Parse '0.0,0.5,1.0' -> [0.0,0.5,1.0]."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


Tau = _parse_tau(tau_str)
grid = np.linspace(0.0, float(Tau[-1]), int(grid_n))

st.divider()

# -------------------------
# 3) Options d'affichage de la progression
# -------------------------
# pfe_profile_swap a été patché pour accepter un progress_cb optionnel,
# avec un mode détaillé (par cashflow) quand la schedule est longue.
st.subheader("Run controls")

colA, colB, colC = st.columns([1.0, 1.0, 1.2])

with colA:
    inner_progress = st.checkbox(
        "Show detailed progress (per cashflow)",
        value=False,
        help="Affiche une progression plus fine (à l'intérieur de chaque point de grille). Peut être un peu plus lent côté UI.",
    )

with colB:
    inner_every = st.number_input(
        "Update frequency (cashflows)",
        value=3,
        min_value=1,
        step=1,
        help="Si le mode détaillé est activé: update toutes les N cashflows.",
    )

with colC:
    show_table = st.checkbox(
        "Show progress table",
        value=True,
        help="Affiche un mini tableau des derniers updates (utile pour diagnostiquer si ça bloque).",
    )

status_ph = st.empty()
bar_ph = st.empty()
table_ph = st.empty()


def _run_with_progress(curve_sim, title: str):
    """
    Lance pfe_profile_swap en branchant un callback Streamlit pour :
      - barre de progression
      - statut texte (grid/cashflows)
      - mini table des events
    """
    st.session_state["pfe_progress_rows"] = []
    prog = bar_ph.progress(0, text="Initialisation Monte Carlo...")

    def progress_cb(d: dict):
        rows = st.session_state.get("pfe_progress_rows", [])
        rows.append(d)
        st.session_state["pfe_progress_rows"] = rows

        stage = d.get("stage", "grid")
        pct = d.get("pct", None)

        if pct is not None:
            pct_int = int(max(0, min(100, round(100 * float(pct)))))
            prog.progress(pct_int, text=f"{title} — {pct_int}%")
        else:
            prog.progress(0, text=f"{title} — ...")

        if stage == "cashflows":
            gi = d.get("grid_i", "?")
            gn = d.get("grid_n", "?")
            cf_i = d.get("cf_i", "?")
            cf_n = d.get("cf_n", "?")
            t = d.get("t", None)
            Ti = d.get("Ti", None)
            status_ph.markdown(
                f"**{title} — simulation en cours**  \n"
                f"- Grid: **{gi}/{gn}** (t={float(t):.4f})  \n"
                f"- Cashflows: **{cf_i}/{cf_n}** (Ti={float(Ti):.4f})"
            )
        else:
            gi = d.get("grid_i", "?")
            gn = d.get("grid_n", "?")
            t = d.get("t", None)
            pfe_t = d.get("pfe_t", None)
            epe_t = d.get("epe_t", None)
            status_ph.markdown(
                f"**{title} — grid {gi}/{gn}** (t={float(t):.4f})  \n"
                f"- PFE={float(pfe_t):,.0f}  |  EPE={float(epe_t):,.0f}"
            )

        if show_table and rows:
            tail = pd.DataFrame(rows).tail(25)
            table_ph.dataframe(tail, use_container_width=True, hide_index=True, height=240)

    pfe, epe = pfe_profile_swap(
        curve_sim=curve_sim,
        grid=grid,
        Tau=Tau,
        K=float(K),
        N=float(notional),
        payer=bool(payer),
        q=float(q),
        progress_cb=progress_cb,
        inner_progress=bool(inner_progress),
        inner_every=int(inner_every),
    )
    prog.progress(100, text=f"{title} — done ✅")
    return pfe, epe


# -------------------------
# 4) Reconstruction d'un "curve_sim" à partir du run sélectionné
# -------------------------
# Ici on évite TOTALEMENT la recalibration :
#   - HW1F: on construit un HullWhitePricer avec params + curve -> curve_sim (HullWhiteCurveBuilder)
#   - HW2F: on construit un HullWhite2FPricer (analytique) + un HW2FCurveSim (MC sur x,y)
def _build_curve_sim_from_selected_run():
    if selected_run is None:
        return None, None

    model = (run_model or "").strip()

    # --- Courbe ---
    # DB : déjà reconstruite via curve_from_dict
    # Session : on essaie (a) snapshot de courbe dans last_run, sinon (b) pricer en session_state
    if run_curve is not None:
        curve = run_curve
    else:
        snap = selected_run.get("curve", None)
        if snap:
            curve = curve_from_dict(snap)
        else:
            if model == "HW1F" and "hw1f_pricer" in st.session_state:
                curve = st.session_state["hw1f_pricer"].curve
            elif model == "HW2F" and "hw2f_pricer" in st.session_state:
                curve = st.session_state["hw2f_pricer"].curve
            else:
                st.error("Impossible de reconstruire la courbe: ni snapshot `curve`, ni pricer en session.")
                return None, None

    # --- Paramètres ---
    params = run_params or selected_run.get("params", None)
    if not isinstance(params, dict):
        st.error("Paramètres modèle manquants/incohérents dans le run.")
        return None, None

    # --- Construction simulateur selon modèle ---
    if model == "HW1F":
        # On (re)crée un pricer 1F uniquement pour récupérer curve_sim (et le model params)
        pricer = HullWhitePricer(
            curve,
            n_paths=int(n_paths),
            n_steps=int(n_steps_1f),
            seed=int(seed),
            hw_params=params,
        )
        return pricer.curve_sim, {"curve": curve, "model": "HW1F", "params": pricer.model.parameters}

    if model == "HW2F":
        # Pricer 2F -> fournit model (paramètres + fonctions fermées)
        pricer2 = HullWhite2FPricer(curve, hw2f_params=params)

        # Simu 2F : distribution de P(t,T) via tirages (x_t, y_t)
        curve_sim_2f = HW2FCurveSim(
            curve=curve,
            model=pricer2.model,
            n_paths=int(n_paths),
            seed=int(seed),
            use_legacy_global_seed=True, 
        )
        return curve_sim_2f, {"curve": curve, "model": "HW2F", "params": pricer2.model.parameters}

    st.error(f"Model inconnu dans le run: {model!r}")
    return None, None


# -------------------------
# 5) Bouton "Run" + affichage résultats
# -------------------------
do_run = st.button("🚀 Run PFE (no recalibration)", type="primary", use_container_width=True)

if do_run:
    curve_sim, info = _build_curve_sim_from_selected_run()
    if curve_sim is None:
        st.stop()

    st.subheader("Run info")
    st.json(
        {
            "model": info["model"],
            "params": info["params"],
            "mc": {"n_paths": int(n_paths), "seed": int(seed), "n_steps_1f": int(n_steps_1f)},
            "swap": {"payer": bool(payer), "N": float(notional), "K": float(K), "Tau": Tau},
            "grid_n": int(grid_n),
            "q": float(q),
        }
    )

    # Lancement PFE/EPE + UI progress
    with st.spinner("PFE en cours..."):
        title = f"{info['model']} | PFE swap"
        pfe, epe = _run_with_progress(curve_sim, title=title)

    st.success("Terminé ✅", icon="✅")

    # Affichage des résultats + export
    st.subheader("Results")
    df_out = pd.DataFrame({"t": grid, "PFE": pfe, "EPE": epe})
    st.dataframe(df_out, use_container_width=True, height=260)

    st.pyplot(fig_pfe(grid, pfe, epe=epe, q=float(q), title=title), clear_figure=True)

    st.download_button(
        "⬇️ Download results (CSV)",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="pfe_swap.csv",
        mime="text/csv",
        use_container_width=True,
    )
