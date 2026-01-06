from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from ir.market.loaders_excel import (
    load_curve_xlsx,
    load_swaption_template_xlsx,
    load_caplet_template_xlsx,
)


# -----------------------------------------------------------------------------
# Objectif de ce module
# -----------------------------------------------------------------------------
# Ce fichier centralise le Streamlit pour :
# - récupérer un fichier .xlsx uploadé via st.file_uploader (objet en mémoire)
# - l’écrire en fichier temporaire sur disque (car pandas.read_excel attend un path)
# - appeler les loaders Excel "propres" (ir.market.loaders_excel) pour construire :
#     * Curve
#     * SwaptionQuoteSet
#     * CapletQuoteSet
#
# Points importants :
# - On cache l’écriture disque via st.cache_data pour éviter de réécrire le même upload
#   à chaque rerun Streamlit.
# - Le fichier temporaire est créé avec delete=False : il reste sur disque après fermeture.
# -----------------------------------------------------------------------------


@dataclass
class UploadedXlsx:
    """
    Petit conteneur "propre" pour tracer un upload sauvegardé en temporaire.

    Attributes
    ----------
    name : str
        Nom original du fichier uploadé (filename côté client).
    path : str
        Chemin local du fichier temporaire sur disque.
    """
    name: str
    path: str


@st.cache_data(show_spinner=False)
def _write_tmp_xlsx(file_bytes: bytes, filename: str) -> UploadedXlsx:
    """
    Écrit les bytes d’un fichier uploadé dans un fichier temporaire.

    """
    suffix = Path(filename).suffix if filename else ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        # On renvoie un objet "UploadedXlsx" avec le nom + le chemin du tmp
        return UploadedXlsx(name=filename, path=tmp.name)


def load_curve_and_swaption_from_upload(
    uploaded,
    *,
    curve_sheet: str = "Curve",
    template_sheet: str = "Template",
    smooth: float = 1e-7,
):
    """
    Chargement standard calibration swaptions :
    - écrire l’upload sur disque (tmp)
    - lire la courbe (sheet curve_sheet)
    - lire le template swaption (sheet template_sheet)

    Returns
    -------
    (source_name, curve, swpn)
        source_name : str (nom du fichier uploadé)
        curve : Curve
        swpn : SwaptionQuoteSet
    """
    file_bytes = uploaded.getvalue()
    tmp = _write_tmp_xlsx(file_bytes, uploaded.name)

    curve = load_curve_xlsx(tmp.path, sheet=curve_sheet, smooth=smooth)
    swpn = load_swaption_template_xlsx(tmp.path, sheet=template_sheet)
    return tmp.name, curve, swpn


def load_curve_only_from_upload(uploaded, *, curve_sheet: str = "Curve", smooth: float = 1e-7):
    """
    Chargement "curve only" :
    utile si une page a seulement besoin de la courbe (sans template swaption/caplet).

    Returns
    -------
    (source_name, curve)
    """
    file_bytes = uploaded.getvalue()
    tmp = _write_tmp_xlsx(file_bytes, uploaded.name)
    curve = load_curve_xlsx(tmp.path, sheet=curve_sheet, smooth=smooth)
    return tmp.name, curve


def load_caplet_template_from_upload(uploaded, *, sheet: str = "Template"):
    """
    Chargement d’un template caplets (calibration caps/caplets) depuis un upload.

    Returns
    -------
    (source_name, caplets)
        caplets : CapletQuoteSet
    """
    file_bytes = uploaded.getvalue()
    tmp = _write_tmp_xlsx(file_bytes, uploaded.name)
    caplets = load_caplet_template_xlsx(tmp.path, sheet=sheet)
    return tmp.name, caplets
