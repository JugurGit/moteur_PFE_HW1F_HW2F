from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.ui.db import list_runs, save_run, get_run, delete_run

st.markdown("# Portfolio Tracking")
st.caption("Historique des calibrations/PFE (local SQLite). Comparaison simple et export.")

last_run = st.session_state.get("last_run", None)
last_pfe = st.session_state.get("last_pfe", None)

colA, colB = st.columns([1.0, 1.2], gap="large")

with colA:
    st.subheader("Save current session run")
    if last_run is None:
        st.info("Aucun run en session (calibre d’abord HW1F/HW2F).", icon="ℹ️")
    else:
        default_label = f"{last_run['model']} | {last_run.get('source_file','')}"
        label = st.text_input("Label", value=default_label)
        notes = st.text_area("Notes", value="", height=120)

        artifacts = dict(last_run.get("artifacts", {}) or {})
        if last_pfe is not None:
            artifacts["pfe"] = last_pfe

        # IMPORTANT: curve snapshot (sinon la DB ne peut pas stocker correctement un run)
        curve_snapshot = last_run.get("curve_snapshot", None) or st.session_state.get("last_curve_snapshot", None)

        if st.button("💾 Save run", type="primary", use_container_width=True):
            if curve_snapshot is None:
                st.error(
                    "Impossible de sauver : curve_snapshot absent. "
                    "Relance une calibration (HW1F/HW2F) après avoir appliqué le patch.",
                    icon="⚠️",
                )
            else:
                run_id = save_run(
                    label=label,
                    model=last_run["model"],
                    source_file=last_run.get("source_file"),
                    rmsre=last_run.get("rmsre"),
                    curve_snapshot=curve_snapshot,
                    params=last_run.get("params", {}),
                    artifacts=artifacts,
                    notes=notes,
                    meta={},  # extensible
                )
                st.success(f"Saved ✅ (id={run_id})", icon="✅")
                st.rerun()

with colB:
    st.subheader("Runs history")

    runs = list_runs()
    df = pd.DataFrame(runs)
    st.dataframe(df, use_container_width=True, height=260)

    if df.empty:
        st.stop()

    ids = df["id"].astype(int).tolist()
    pick1 = st.selectbox("Select run A", ids, index=0)
    pick2 = st.selectbox("Select run B (optional)", [None] + ids, index=0)

    runA = get_run(int(pick1))
    st.markdown("### Run A")
    st.json({k: runA.get(k) for k in ["id", "created_at", "label", "model", "source_file", "rmsre"]})
    st.json(runA.get("params", {}))

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
    if st.button("🗑️ Delete run A"):
        delete_run(int(pick1))
        st.warning("Deleted run A.", icon="🗑️")
        st.rerun()
