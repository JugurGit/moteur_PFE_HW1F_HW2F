import streamlit as st

st.markdown("# Overview")
st.write(
    "Cette app expose :\n"
    "- Calibration Hull–White **1F** (a, sigma)\n"
    "- Calibration Hull–White **2F (G2++)** (a,b,rho) + inner (sigma,eta)\n"
    "- **PFE / EPE** swap via Monte Carlo (1F et 2F)\n"
    "- **Portfolio tracking** : historique de runs, comparaison, export\n"
    "- **Documentation** : navigation dans le code"
)

st.success("Astuce : commencer par `Calibration HW1F` ou `Calibration HW2F`, puis aller sur `PFE Swap`.", icon="✅")
