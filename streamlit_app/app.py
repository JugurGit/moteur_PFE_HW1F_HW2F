from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is importable (so "import ir...." works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.ui.db import init_db

st.set_page_config(
    page_title="IR Lab | Hull-White",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# --- Global sidebar ---
with st.sidebar:
    st.markdown("## IR Lab")
    st.caption("Hull–White 1F / 2F • Calibration • PFE • Tracking")

    tracking = st.toggle("📌 Portfolio tracking mode", value=True)
    st.session_state["tracking_mode"] = tracking

    st.divider()

# --- Page content (marketing / mémoire) ---

st.title("📈 PFE d’un swap IRS 5Y sous Hull–White 1F et Hull–White 2F")
st.caption("Hull–White 1F / 2F • Calibration • PFE • Portfolio Tracking • Démo associé au mémoire")

st.markdown("### 🧩 Contexte — Du mémoire à la démo")

st.info(
    """
Ce projet est la **démo technique** associé à mon mémoire : il recrée, dans un cadre **pédagogique et reproductible**,
un workflow de modélisation taux utilisé en pratique pour **industrialiser** des calculs et produire des résultats **traçables**.

L’application couvre la chaîne **courbe → pricing → calibration Hull–White 1F/2F → simulation → exposition**,
avec un module de **Portfolio Tracking** pour historiser et comparer les runs.
""",
    icon="📌",
)

st.warning(
    """
Je ne dispose pas des **données internes** ni de la **documentation** nécessaires
pour illustrer les traitements de manière “réelle”.
Le projet remplace donc ces entrées par des données **contrôlées / simulées**, tout en conservant
la **structuration** et l’**auditabilité** attendues dans un environnement professionnel.
""",
    icon="⚠️",
)

st.markdown("### 🎯 Ce que démontre ce mini-projet (workflow end-to-end)")

cA, cB, cC, cD = st.columns(4)
with cA:
    st.markdown("**1) Inputs maîtrisés**")
    st.caption("Courbes • instruments • paramètres • scénarios")
with cB:
    st.markdown("**2) Calibration HW 1F/2F**")
    st.caption("Fits par tenors • diagnostics • comparaisons")
with cC:
    st.markdown("**3) Risque d’exposition**")
    st.caption("Simulation • EPE/PFE swap • profils temporels")
with cD:
    st.markdown("**4) Tracking & rejouabilité**")
    st.caption("Runs historisés • comparaisons • restauration")

st.success(
    """
**En résumé** : un labo orienté **production** (calibration + exposition) qui matérialise le cœur du mémoire :
des calculs **reproductibles**, **comparables** et **auditables**, présentés via une UI claire et “reporting-ready”.
""",
    icon="✅",
)



st.divider()



