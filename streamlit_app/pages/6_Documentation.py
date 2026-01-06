from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

# Panneau de doc "manuel" + registry JSON 
from streamlit_app.ui.code_docs import render_doc_panel, load_docs_registry

ROOT = Path(__file__).resolve().parents[2]

# Extensions que l’on sait "prévisualiser proprement" 
TEXT_EXTS = {
    ".py": "python",
    ".ipynb": "ipynb",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
}

# Limite de preview (pour éviter de faire exploser la mémoire / UI)
MAX_PREVIEW_BYTES = 1_200_000  # ~1.2MB

# Options UI 
SHOW_META = True
WRAP_LINES = False  


# -------------------------
# Helpers formatting
# -------------------------
def _fmt_bytes(n: int) -> str:
    """Affichage lisible d'une taille en octets."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    return f"{n/(1024*1024):.2f} MB"


def _fmt_dt(ts: float) -> str:
    """Affichage lisible d'un timestamp (mtime)."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _read_text_safely(path: Path, max_bytes: int = MAX_PREVIEW_BYTES) -> Tuple[str, bool]:
    """
    Lit un fichier en bytes puis décode UTF-8 (avec replacement).
    Retourne (text, truncated) où truncated=True si on a coupé à max_bytes.
    """
    b = path.read_bytes()
    truncated = False
    if len(b) > max_bytes:
        b = b[:max_bytes]
        truncated = True
    text = b.decode("utf-8", errors="replace")
    return text, truncated


# -------------------------
# Notebook rendering (.ipynb)
# -------------------------
def _render_notebook(path: Path) -> None:
    """
    Rend un .ipynb en mode "lecture" :
      - cellules markdown via st.markdown
      - cellules code via st.code
    """
    raw, truncated = _read_text_safely(path)
    if truncated:
        st.warning(
            f"Notebook trop volumineux : preview tronquée à {_fmt_bytes(MAX_PREVIEW_BYTES)}. "
            "Tu peux le télécharger pour l’ouvrir complet.",
            icon="⚠️",
        )

    # Le .ipynb est du JSON
    try:
        nb = json.loads(raw)
    except Exception:
        st.error("Impossible de parser ce .ipynb (JSON invalide).", icon="❌")
        st.code(raw, language="json")
        return

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        st.error("Format .ipynb inattendu (cells manquant).", icon="❌")
        st.code(raw, language="json")
        return

    # Toggle simple pour éviter de spammer l’UI
    show_outputs = st.checkbox("Afficher outputs", value=False)
    st.divider()

    for i, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type", "")
        src = cell.get("source", "")

        if isinstance(src, list):
            src_text = "".join(src)
        else:
            src_text = str(src)

        if cell_type == "markdown":
            if src_text.strip():
                st.markdown(src_text)

        elif cell_type == "code":
            st.markdown(f"**In [{i}]**")
            st.code(src_text, language="python")

            if show_outputs:
                outs = cell.get("outputs", [])
                if isinstance(outs, list) and outs:
                    for out in outs:
                        otype = out.get("output_type", "")

                        if otype == "stream":
                            txt = out.get("text", "")
                            if isinstance(txt, list):
                                txt = "".join(txt)
                            st.code(str(txt), language="text")

                        elif otype in ("execute_result", "display_data"):
                            data = out.get("data", {})
                            txt = None
                            if isinstance(data, dict):
                                txt = data.get("text/plain", None)
                            if txt is not None:
                                if isinstance(txt, list):
                                    txt = "".join(txt)
                                st.code(str(txt), language="text")

                        elif otype == "error":
                            tb = out.get("traceback", [])
                            if isinstance(tb, list):
                                st.code("\n".join(tb), language="text")

        else:
            continue


def _language_for(path: Path) -> str:
    """Retourne le langage (string) pour st.code en fonction de l'extension."""
    ext = path.suffix.lower()
    if ext == ".ipynb":
        return "json"
    return TEXT_EXTS.get(ext, "text")


def _meta_for_rel(rel: str) -> Optional[Dict]:
    """
    Construit une petite fiche meta sur un fichier relpath depuis ROOT.
    Retourne None si:
      - fichier inexistant
      - extension non supportée
      - erreur I/O
    """
    p = ROOT / rel
    if not p.exists() or not p.is_file():
        return None

    ext = p.suffix.lower()
    if ext not in TEXT_EXTS:
        return None

    try:
        stt = p.stat()
        size = int(stt.st_size)
        mtime = float(stt.st_mtime)
    except OSError:
        return None

    n_lines: Optional[int] = None
    if ext != ".ipynb" and size <= 400_000:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            n_lines = txt.count("\n") + 1 if txt else 0
        except Exception:
            n_lines = None

    return {
        "rel": rel,
        "path": str(p),
        "ext": ext,
        "size": size,
        "mtime": mtime,
        "lines": n_lines,
    }


# =============================
# UI
# =============================
st.markdown("# Documentation")
st.caption("Affiche la documentation des fichiers principaux du projet.")

# 1) On charge la registry (JSON) : c’est la source de vérité des fichiers documentés
registry = load_docs_registry()
registry_rels = sorted([k for k in registry.keys() if isinstance(k, str) and k.strip()])

if not registry_rels:
    st.error("docs_registry.json est vide (aucun fichier référencé).", icon="❌")
    st.stop()

all_files: List[Dict] = []
missing: List[str] = []
skipped: List[str] = []

for rel in registry_rels:
    m = _meta_for_rel(rel)
    if m is None:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        else:
            skipped.append(rel)
        continue
    all_files.append(m)

# Messages "diagnostic" si registry contient des entrées non affichables
if missing:
    st.warning(
        "Certains fichiers sont référencés dans docs_registry.json mais introuvables sur disque :\n"
        + "\n".join([f"- {x}" for x in missing]),
        icon="⚠️",
    )

if skipped:
    st.info(
        "Certains fichiers référencés sont ignorés (extension non supportée pour preview) :\n"
        + "\n".join([f"- {x}" for x in skipped]),
        icon="ℹ️",
    )

if not all_files:
    st.error("Aucun fichier affichable (tous manquants ou ignorés).", icon="❌")
    st.stop()

rels = [d["rel"] for d in all_files]

# -------------------------
# Sélection + navigation (Prev / Next)
# -------------------------
if "px_choice" not in st.session_state:
    st.session_state["px_choice"] = rels[0]
if st.session_state["px_choice"] not in rels:
    st.session_state["px_choice"] = rels[0]

nav1, nav2, nav3, nav4 = st.columns([0.18, 0.18, 1.0, 0.28], gap="small")

with nav1:
    if st.button("⬅️ Prev", use_container_width=True):
        i = rels.index(st.session_state["px_choice"])
        st.session_state["px_choice"] = rels[max(0, i - 1)]

with nav2:
    if st.button("Next ➡️", use_container_width=True):
        i = rels.index(st.session_state["px_choice"])
        st.session_state["px_choice"] = rels[min(len(rels) - 1, i + 1)]

with nav4:
    st.caption(f"{rels.index(st.session_state['px_choice']) + 1} / {len(rels)}")

choice = st.selectbox(
    "Select file",
    rels,
    index=rels.index(st.session_state["px_choice"]),
)
st.session_state["px_choice"] = choice

path = ROOT / choice
meta = next(d for d in all_files if d["rel"] == choice)

# -------------------------
# Header + actions
# -------------------------
st.write(f"**{choice}**")

if SHOW_META:
    st.caption(
        f"Ext: `{meta['ext']}`  |  Size: {_fmt_bytes(meta['size'])}  |  "
        f"Modified: {_fmt_dt(meta['mtime'])}"
        + (f"  |  Lines: {meta['lines']}" if meta.get("lines") is not None else "")
    )

# Bouton de download (utile si preview tronquée)
try:
    file_bytes = path.read_bytes()
    st.download_button(
        "⬇️ Download file",
        data=file_bytes,
        file_name=path.name,
        mime="text/plain",
        use_container_width=False,
    )
except Exception:
    st.warning("Impossible de préparer le téléchargement (droits/IO).", icon="⚠️")

# -------------------------
# Preview + doc panel
# -------------------------
ext = path.suffix.lower()

# Layout : à gauche preview, à droite doc manuelle (docs_registry + annotations)
left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.subheader("Preview")

    # Cas notebook : soit rendu, soit JSON brut
    if ext == ".ipynb":
        view_mode = st.radio("Notebook view", ["Rendered", "Raw JSON"], horizontal=True, index=0)
        if view_mode == "Rendered":
            _render_notebook(path)
        else:
            raw, truncated = _read_text_safely(path)
            if truncated:
                st.warning(
                    f"Preview tronquée à {_fmt_bytes(MAX_PREVIEW_BYTES)} (notebook trop volumineux).",
                    icon="⚠️",
                )
            st.code(raw, language="json")

    # Cas fichiers texte : preview direct
    else:
        try:
            content, truncated = _read_text_safely(path)
            if truncated:
                st.warning(
                    f"Preview tronquée à {_fmt_bytes(MAX_PREVIEW_BYTES)} (fichier volumineux).",
                    icon="⚠️",
                )

            lang = _language_for(path)

            st.code(content, language=lang)

        except Exception as e:
            st.error(f"Erreur de lecture : {e}", icon="❌")

with right:
    render_doc_panel(choice, path)
