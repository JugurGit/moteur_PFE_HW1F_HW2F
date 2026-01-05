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
    st.markdown("### Quick tips")
    st.markdown(
        "- Calibre dans **HW1F** ou **HW2F**\n"
        "- Va dans **PFE Swap** pour lancer l’expo\n"
        "- Sauvegarde un run dans **Portfolio Tracking**"
    )

st.title("📈 IR Lab — Hull–White Playground")
st.write(
    "Utilise le menu de gauche (pages) pour naviguer : calibration 1F/2F, PFE, tracking, explorer."
)

st.info(
    "Pages disponibles : Overview • Calibration HW1F • Calibration HW2F • PFE Swap • Portfolio Tracking • Project Explorer",
    icon="ℹ️",
)
