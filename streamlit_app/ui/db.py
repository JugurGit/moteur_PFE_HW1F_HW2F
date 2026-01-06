# streamlit_app/ui/db.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Objectif de ce module
# -----------------------------------------------------------------------------
# Ce fichier fournit une mini couche "persistence" via SQLite pour :
# - sauvegarder des runs (calibration HW1F/HW2F, PFE, artefacts, etc.)
# - relister / relire / supprimer des runs
# - sérialiser/désérialiser la courbe (Curve) afin de pouvoir reconstruire un run
#
# Points clés :
# - DB locale dans streamlit_app/data/irlab.db
# - list_runs est cache_data pour accélérer l’UI => on invalide le cache après écriture.
# -----------------------------------------------------------------------------

# Project root: .../streamlit_app/ui/db.py -> parents[2] = project root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "streamlit_app" / "data"
DB_PATH = DATA_DIR / "irlab.db"


def _utc_now_iso() -> str:
    """Timestamp ISO en UTC (secondes) pour stocker un created_at stable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    """
    Ouvre une connexion SQLite.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Retourne l’ensemble des colonnes d’une table (via PRAGMA table_info)."""
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return {str(r["name"]) for r in rows}


def init_db() -> None:
    """
    Initialise la base (idempotent) et applique des "soft migrations".

    - Crée la table runs si absente
    - Ajoute des colonnes manquantes si DB plus ancienne
    - Crée quelques index utiles pour l’UI
    """
    conn = _connect()
    cur = conn.cursor()

    # Table de base 
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            label TEXT,
            model TEXT NOT NULL,                 -- "HW1F" / "HW2F"
            source_file TEXT,                    -- nom du xlsx uploadé (info)
            rmsre REAL,
            curve_json TEXT NOT NULL,            -- snapshot Curve sérialisé
            params_json TEXT NOT NULL,           -- paramètres modèle sérialisés
            artifacts_json TEXT,                 -- optionnel: PFE, etc.
            notes TEXT,                          -- optionnel: notes user
            meta_json TEXT                        -- optionnel: metadata extensible
        )
        """
    )

    cols = _table_columns(conn, "runs")

    def _add_col(name: str, ddl: str):
        if name not in cols:
            cur.execute(f"ALTER TABLE runs ADD COLUMN {ddl}")

    _add_col("label", "label TEXT")
    _add_col("artifacts_json", "artifacts_json TEXT")
    _add_col("notes", "notes TEXT")
    _add_col("meta_json", "meta_json TEXT")

    # Index (améliore la vitesse des listages/filtrages)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model)")
    conn.commit()
    conn.close()


# -------------------------
# Curve serialization helpers
# -------------------------

def curve_to_dict(curve) -> dict:
    """
    Snapshot minimal pour reconstruire ir.market.curve.Curve.

    Hypothèses:
    - curve.time : array-like de maturités (années)
    - curve.discount_factors : array-like de DF
    - curve.smooth : paramètre de lissage spline (optionnel)
    """
    return {
        "time": [float(x) for x in getattr(curve, "time")],
        "discount_factors": [float(x) for x in getattr(curve, "discount_factors")],
        "smooth": float(getattr(curve, "smooth", 1e-7)),
    }


def curve_from_dict(d: dict):
    """
    Reconstruit un objet Curve à partir du snapshot dict.
    Import local pour éviter les soucis d’import au démarrage Streamlit.
    """
    from ir.market.curve import Curve  

    time = d.get("time", None)
    disc = d.get("discount_factors", None)
    smooth = d.get("smooth", 1e-7)

    if time is None or disc is None:
        raise ValueError("Invalid curve snapshot: missing 'time' or 'discount_factors'.")

    return Curve(time, disc, smooth=float(smooth))


# -------------------------
# Cache helpers
# -------------------------

def _clear_runs_cache() -> None:
    """
    Invalidation du cache list_runs (Streamlit cache_data).
    """
    try:
        list_runs.clear()  # type: ignore[attr-defined]
    except Exception:
        # fallback global
        try:
            st.cache_data.clear()
        except Exception:
            pass


# -------------------------
# Runs API
# -------------------------

def save_run(
    *,
    model: str,
    curve_snapshot: dict,
    params: dict,
    source_file: Optional[str] = None,
    rmsre: Optional[float] = None,
    label: Optional[str] = None,
    artifacts: Optional[dict] = None,
    notes: Optional[str] = None,
    meta: Optional[dict] = None,
) -> int:
    """
    Insère un run en base et retourne son id.

    """
    model = str(model).strip()
    if model not in ("HW1F", "HW2F"):
        raise ValueError("Le modèle doit être be 'HW1F' ou 'HW2F'.")

    if curve_snapshot is None:
        raise ValueError(
            "curve_snapshot est requis (run une première calibration, ou assurer que last_curve_snapshot is validé)."
        )

    created_at = _utc_now_iso()

    # Sérialisation JSON "safe" (UTF-8)
    curve_json = json.dumps(curve_snapshot, ensure_ascii=False)
    params_json = json.dumps(params or {}, ensure_ascii=False)
    artifacts_json = json.dumps(artifacts or {}, ensure_ascii=False)
    notes_txt = "" if notes is None else str(notes)

    meta_out = dict(meta or {})
    if label is not None:
        meta_out.setdefault("label", str(label))
    if artifacts is not None:
        meta_out.setdefault("artifacts", artifacts)
    if notes is not None:
        meta_out.setdefault("notes", notes_txt)
    meta_json = json.dumps(meta_out, ensure_ascii=False)

    conn = _connect()
    cols = _table_columns(conn, "runs")
    cur = conn.cursor()

    ins_cols = ["created_at", "model", "source_file", "rmsre", "curve_json", "params_json"]
    ins_vals = [created_at, model, source_file, rmsre, curve_json, params_json]

    if "label" in cols:
        ins_cols.append("label")
        ins_vals.append(label)
    if "artifacts_json" in cols:
        ins_cols.append("artifacts_json")
        ins_vals.append(artifacts_json)
    if "notes" in cols:
        ins_cols.append("notes")
        ins_vals.append(notes_txt)
    if "meta_json" in cols:
        ins_cols.append("meta_json")
        ins_vals.append(meta_json)

    placeholders = ",".join(["?"] * len(ins_cols))
    colnames = ",".join(ins_cols)

    cur.execute(
        f"""
        INSERT INTO runs({colnames})
        VALUES ({placeholders})
        """,
        tuple(ins_vals),
    )
    conn.commit()
    run_id = int(cur.lastrowid)
    conn.close()

    _clear_runs_cache()
    return run_id


@st.cache_data(show_spinner=False)
def list_runs(limit: int = 200) -> list[dict]:
    """
    Liste les runs (ordre décroissant created_at) et retourne une liste de dicts.

    Important:
    - Cache Streamlit : accélère le rendu UI
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM runs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()

    out: list[dict] = []
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r["meta_json"]) if ("meta_json" in r.keys() and r["meta_json"]) else {}
        except Exception:
            meta = {}

        label = None
        if "label" in r.keys():
            label = r["label"]
        if not label:
            label = meta.get("label", None)

        artifacts = {}
        if "artifacts_json" in r.keys() and r["artifacts_json"]:
            try:
                artifacts = json.loads(r["artifacts_json"])
            except Exception:
                artifacts = {}
        else:
            artifacts = meta.get("artifacts", {}) or {}

        notes = ""
        if "notes" in r.keys() and r["notes"]:
            notes = str(r["notes"])
        else:
            notes = str(meta.get("notes", ""))

        out.append(
            {
                "id": int(r["id"]),
                "created_at": str(r["created_at"]),
                "label": label,
                "model": str(r["model"]),
                "source_file": r["source_file"],
                "rmsre": r["rmsre"],
                "curve": json.loads(r["curve_json"]) if r["curve_json"] else None,
                "params": json.loads(r["params_json"]) if r["params_json"] else {},
                "artifacts": artifacts,
                "notes": notes,
                "meta": meta,
            }
        )
    return out


def get_run(run_id: int) -> Optional[dict]:
    """
    Récupère un run par id (retourne le même format que list_runs, mais pour 1 seul).
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs WHERE id = ?", (int(run_id),))
    r = cur.fetchone()
    conn.close()

    if r is None:
        return None

    meta = {}
    try:
        meta = json.loads(r["meta_json"]) if ("meta_json" in r.keys() and r["meta_json"]) else {}
    except Exception:
        meta = {}

    label = None
    if "label" in r.keys():
        label = r["label"]
    if not label:
        label = meta.get("label", None)

    artifacts = {}
    if "artifacts_json" in r.keys() and r["artifacts_json"]:
        try:
            artifacts = json.loads(r["artifacts_json"])
        except Exception:
            artifacts = {}
    else:
        artifacts = meta.get("artifacts", {}) or {}

    notes = ""
    if "notes" in r.keys() and r["notes"]:
        notes = str(r["notes"])
    else:
        notes = str(meta.get("notes", ""))

    return {
        "id": int(r["id"]),
        "created_at": str(r["created_at"]),
        "label": label,
        "model": str(r["model"]),
        "source_file": r["source_file"],
        "rmsre": r["rmsre"],
        "curve": json.loads(r["curve_json"]) if r["curve_json"] else None,
        "params": json.loads(r["params_json"]) if r["params_json"] else {},
        "artifacts": artifacts,
        "notes": notes,
        "meta": meta,
    }


def delete_run(run_id: int) -> bool:
    """
    Supprime un run par id.

    Returns
    -------
    bool
        True si au moins une ligne supprimée, False sinon.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
    conn.commit()
    deleted = (cur.rowcount or 0) > 0
    conn.close()

    _clear_runs_cache()
    return deleted


def format_run_label(run: dict) -> str:
    """
    Formate une ligne "lisible" pour afficher un run dans un selectbox Streamlit.
    """
    rid = run.get("id", "?")
    ts = run.get("created_at", "")
    model = run.get("model", "?")
    src = run.get("source_file") or "unknown"
    label = run.get("label", None)
    rmsre = run.get("rmsre", None)

    head = f"#{rid} | {model} | {ts}"
    if label:
        head += f" | {label}"
    if rmsre is not None:
        head += f" | RMSRE={float(rmsre):.2e}"
    head += f" | {src}"
    return head
