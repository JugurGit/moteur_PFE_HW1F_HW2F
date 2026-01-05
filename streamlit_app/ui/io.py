from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from ir.market.loaders_excel import load_curve_xlsx, load_swaption_template_xlsx, load_caplet_template_xlsx


@dataclass
class UploadedXlsx:
    name: str
    path: str

@st.cache_data(show_spinner=False)
def _write_tmp_xlsx(file_bytes: bytes, filename: str) -> UploadedXlsx:
    suffix = Path(filename).suffix if filename else ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        return UploadedXlsx(name=filename, path=tmp.name)

def load_curve_and_swaption_from_upload(
    uploaded,
    *,
    curve_sheet: str = "Curve",
    template_sheet: str = "Template",
    smooth: float = 1e-7,
):
    file_bytes = uploaded.getvalue()
    tmp = _write_tmp_xlsx(file_bytes, uploaded.name)

    curve = load_curve_xlsx(tmp.path, sheet=curve_sheet, smooth=smooth)
    swpn = load_swaption_template_xlsx(tmp.path, sheet=template_sheet)
    return tmp.name, curve, swpn

def load_curve_only_from_upload(uploaded, *, curve_sheet: str = "Curve", smooth: float = 1e-7):
    file_bytes = uploaded.getvalue()
    tmp = _write_tmp_xlsx(file_bytes, uploaded.name)
    curve = load_curve_xlsx(tmp.path, sheet=curve_sheet, smooth=smooth)
    return tmp.name, curve

def load_caplet_template_from_upload(uploaded, *, sheet: str = "Template"):
    file_bytes = uploaded.getvalue()
    tmp = _write_tmp_xlsx(file_bytes, uploaded.name)
    caplets = load_caplet_template_xlsx(tmp.path, sheet=sheet)
    return tmp.name, caplets
