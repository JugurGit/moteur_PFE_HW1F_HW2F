# streamlit_app/ui/code_docs.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Objectif de ce module
# -----------------------------------------------------------------------------
# Ce fichier centralise la logique de "documentation manuelle" des fichiers du projet.
# L’idée : un JSON (docs_registry.json) contient, pour chaque fichier (relpath),
# une fiche courte (title / summary / usage / notes / tags).
#
# Cette brique est utilisée par :
# - Documentation (ou pages de navigation/preview)
# - Page Documentation
# - Toute page Streamlit qui veut afficher un panneau de doc à droite
# -----------------------------------------------------------------------------

# Racine du projet (à adapter si la structure change)
ROOT = Path(__file__).resolve().parents[2]

# Emplacement du registry (JSON) : stocké côté app, pas à la racine
REGISTRY_PATH = ROOT / "streamlit_app" / "data" / "docs_registry.json"


def load_docs_registry() -> Dict[str, Any]:
    """
    Charge le fichier docs_registry.json et le retourne sous forme de dict.

    - Clé : relpath du fichier (ex: "ir/pricers/hw1f_pricer.py")
    - Valeur : dict de métadonnées (title/tags/summary/usage/notes)

    Le chargement est mis en cache via st.cache_data car Streamlit rerun souvent.
    """

    @st.cache_data(show_spinner=False)
    def _load(p: str) -> Dict[str, Any]:
        """
        Fonction interne cachée (st.cache_data impose une fonction pure-ish).
        """
        path = Path(p)
        if not path.exists():
            # Si le registry n’existe pas, on retourne un dict vide (pas d’erreur bloquante)
            return {}

        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            # On impose un dict en sortie (sinon on ignore)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            # JSON invalide / erreur de lecture : on évite de casser l’app
            return {}

    return _load(str(REGISTRY_PATH))


def manual_doc_for(relpath: str) -> Optional[Dict[str, Any]]:
    """
    Retourne la fiche de doc manuelle d’un fichier (si présente dans le registry).

    Parameters
    ----------
    relpath : str
        Chemin relatif du fichier (clé dans docs_registry.json).

    Returns
    -------
    dict | None
        dict si trouvé, sinon None.
    """
    reg = load_docs_registry()
    d = reg.get(relpath, None)
    if isinstance(d, dict):
        return d
    return None


def render_doc_panel(relpath: str, path: Path) -> None:
    """
    Affiche le panneau de documentation (côté UI).

    Notes
    -----
    - Ce panneau est volontairement "léger" : pas d’analyse AST, pas d’auto-doc ici.
      Il affiche uniquement ce que tu as écrit dans docs_registry.json.
    - `path` n’est pas utilisé pour le moment
    """
    st.subheader("Documentation")

    manual = manual_doc_for(relpath)

    # Cas où aucune fiche n’existe
    if manual is None:
        st.info(
            "Pas de fiche manuelle pour ce fichier. (Ajoute une entrée dans docs_registry.json)",
            icon="ℹ️",
        )
        return

    # Champs attendus 
    title = manual.get("title", relpath)
    tags = manual.get("tags", [])
    summary = manual.get("summary", "")
    usage = manual.get("usage", "")
    notes = manual.get("notes", "")

    # Rendu UI
    st.markdown(f"### {title}")

    if tags:
        # Affichage compact de tags en monospace
        st.caption(" • ".join([f"`{t}`" for t in tags]))

    if summary:
        st.markdown(summary)

    if usage:
        st.markdown("#### Usage")
        st.markdown(usage)

    if notes:
        st.markdown("#### Notes")
        st.markdown(notes)
