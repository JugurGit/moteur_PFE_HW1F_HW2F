# streamlit_app/ui/code_docs.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# We keep it here so both Project Explorer + other pages can reuse it
ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "streamlit_app" / "data" / "docs_registry.json"


def load_docs_registry() -> Dict[str, Any]:
    """
    Loads manual docs registry (JSON) keyed by file relpath.
    Cached because Streamlit reruns a lot.
    """
    @st.cache_data(show_spinner=False)
    def _load(p: str) -> Dict[str, Any]:
        path = Path(p)
        if not path.exists():
            return {}
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return _load(str(REGISTRY_PATH))


def manual_doc_for(relpath: str) -> Optional[Dict[str, Any]]:
    reg = load_docs_registry()
    d = reg.get(relpath, None)
    if isinstance(d, dict):
        return d
    return None


def render_doc_panel(relpath: str, path: Path) -> None:
    """
    Right-hand panel: manual docs only (Résumé).
    """
    st.subheader("Documentation")

    manual = manual_doc_for(relpath)

    if manual is None:
        st.info(
            "Pas de fiche manuelle pour ce fichier. (Ajoute une entrée dans docs_registry.json)",
            icon="ℹ️",
        )
        return

    title = manual.get("title", relpath)
    tags = manual.get("tags", [])
    summary = manual.get("summary", "")
    usage = manual.get("usage", "")
    notes = manual.get("notes", "")

    st.markdown(f"### {title}")
    if tags:
        st.caption(" • ".join([f"`{t}`" for t in tags]))

    if summary:
        st.markdown(summary)

    if usage:
        st.markdown("#### Usage")
        st.markdown(usage)

    if notes:
        st.markdown("#### Notes")
        st.markdown(notes)
