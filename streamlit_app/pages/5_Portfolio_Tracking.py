from __future__ import annotations

import pandas as pd
import streamlit as st

# DB helpers (SQLite local) :
# - list_runs()   : liste des runs disponibles (métadonnées)
# - save_run(...) : sauvegarde un run (params + curve snapshot + artifacts + notes)
# - get_run(id)   : charge un run complet
# - delete_run(id): supprime un run
from streamlit_app.ui.db import list_runs, save_run, get_run, delete_run

# =============================
# PAGE : PORTFOLIO TRACKING
# =============================
st.markdown("# Portfolio Tracking")
st.caption("Historique des calibrations/PFE (local SQLite). Comparaison simple et export.")

# On récupère ce qui a été produit dans la session par les pages Calibration / PFE
# last_run : dict avec model, params, rmsre, curve_snapshot, etc.
# last_pfe : éventuel artefact PFE (stocké en session quand on lance la page PFE)
last_run = st.session_state.get("last_run", None)
last_pfe = st.session_state.get("last_pfe", None)

# Mise en page : gauche = sauvegarde session ; droite = historique + comparaison
colA, colB = st.columns([1.0, 1.2], gap="large")

# -------------------------
# 1) Sauvegarder le run courant (session -> DB)
# -------------------------
with colA:
    st.subheader("Save current session run")

    # S'il n'y a pas de calibration récente, rien à sauver
    if last_run is None:
        st.info("Aucun run en session (calibre d’abord HW1F/HW2F).", icon="ℹ️")
    else:
        default_label = f"{last_run['model']} | {last_run.get('source_file','')}"
        label = st.text_input("Label", value=default_label)
        notes = st.text_area("Notes", value="", height=120)
        artifacts = dict(last_run.get("artifacts", {}) or {})
        if last_pfe is not None:
            artifacts["pfe"] = last_pfe

        curve_snapshot = last_run.get("curve_snapshot", None) or st.session_state.get("last_curve_snapshot", None)

        if st.button("💾 Save run", type="primary", use_container_width=True):
            if curve_snapshot is None:
                st.error(
                    "Impossible de sauver : curve_snapshot absent. "
                    "Relance une calibration (HW1F/HW2F) après avoir appliqué le patch.",
                    icon="⚠️",
                )
            else:
                # Sauvegarde DB
                run_id = save_run(
                    label=label,
                    model=last_run["model"],
                    source_file=last_run.get("source_file"),
                    rmsre=last_run.get("rmsre"),
                    curve_snapshot=curve_snapshot,
                    params=last_run.get("params", {}),
                    artifacts=artifacts,
                    notes=notes,
                    meta={},  
                )
                st.success(f"Saved ✅ (id={run_id})", icon="✅")
                st.rerun()

# -------------------------
# 2) Historique des runs (DB) + comparaison A/B + suppression
# -------------------------
with colB:
    st.subheader("Runs history")

    # Liste brute des runs (métadonnées) -> dataframe
    runs = list_runs()
    df = pd.DataFrame(runs)
    st.dataframe(df, use_container_width=True, height=260)

    # Si DB vide : on stoppe la page ici
    if df.empty:
        st.stop()

    # Sélection simple (IDs)
    ids = df["id"].astype(int).tolist()
    pick1 = st.selectbox("Select run A", ids, index=0)
    pick2 = st.selectbox("Select run B (optional)", [None] + ids, index=0)

    # --- Run A ---
    runA = get_run(int(pick1))
    st.markdown("### Run A")
    # Affichage résumé (métadonnées clés)
    st.json({k: runA.get(k) for k in ["id", "created_at", "label", "model", "source_file", "rmsre"]})
    # Paramètres modèle (a/sigma/... ou a,b,rho,sigma,eta,...)
    st.json(runA.get("params", {}))

    # --- Run B (optionnel) + diff ---
    if pick2 is not None:
        runB = get_run(int(pick2))
        st.markdown("### Run B")
        st.json({k: runB.get(k) for k in ["id", "created_at", "label", "model", "source_file", "rmsre"]})
        st.json(runB.get("params", {}))

        st.markdown("### Diff (params)")
        keys = sorted(set(runA.get("params", {}).keys()) | set(runB.get("params", {}).keys()))
        rows = []
        for k in keys:
            a = runA.get("params", {}).get(k, None)
            b = runB.get("params", {}).get(k, None)
            rows.append({"param": k, "A": a, "B": b})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

    st.divider()

    # Suppression du run A (action destructive => bouton dédié)
    if st.button("🗑️ Delete run A"):
        delete_run(int(pick1))
        st.warning("Deleted run A.", icon="🗑️")
        st.rerun()
