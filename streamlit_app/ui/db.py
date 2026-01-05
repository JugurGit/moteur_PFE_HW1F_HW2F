# streamlit_app/ui/db.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

# Project root: .../streamlit_app/ui/db.py -> parents[2] = project root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "streamlit_app" / "data"
DB_PATH = DATA_DIR / "irlab.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return {str(r["name"]) for r in rows}


def init_db() -> None:
    conn = _connect()
    cur = conn.cursor()

    # Base table (new installs)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            label TEXT,
            model TEXT NOT NULL,                 -- "HW1F" / "HW2F"
            source_file TEXT,                    -- original uploaded filename (informational)
            rmsre REAL,
            curve_json TEXT NOT NULL,            -- serialized Curve snapshot
            params_json TEXT NOT NULL,           -- serialized model parameters
            artifacts_json TEXT,                 -- optional: PFE, etc.
            notes TEXT,                          -- optional: user notes
            meta_json TEXT                        -- optional: extra metadata
        )
        """
    )

    # Soft migrations for existing DBs created with an older schema
    cols = _table_columns(conn, "runs")

    def _add_col(name: str, ddl: str):
        if name not in cols:
            cur.execute(f"ALTER TABLE runs ADD COLUMN {ddl}")

    _add_col("label", "label TEXT")
    _add_col("artifacts_json", "artifacts_json TEXT")
    _add_col("notes", "notes TEXT")
    _add_col("meta_json", "meta_json TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model)")
    conn.commit()
    conn.close()


# -------------------------
# Curve serialization helpers
# -------------------------

def curve_to_dict(curve) -> dict:
    """
    Minimal snapshot needed to reconstruct ir.market.curve.Curve.
    Assumes curve has attributes: time, discount_factors, smooth.
    """
    return {
        "time": [float(x) for x in getattr(curve, "time")],
        "discount_factors": [float(x) for x in getattr(curve, "discount_factors")],
        "smooth": float(getattr(curve, "smooth", 1e-7)),
    }


def curve_from_dict(d: dict):
    from ir.market.curve import Curve  # local import to avoid early import issues

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
    try:
        list_runs.clear()  # type: ignore[attr-defined]
    except Exception:
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
    # New optional fields (for Portfolio Tracking UI)
    label: Optional[str] = None,
    artifacts: Optional[dict] = None,
    notes: Optional[str] = None,
    meta: Optional[dict] = None,
) -> int:
    """
    Insert a run in DB and return its id.

    Backward compatible:
    - label/artifacts/notes are optional (stored in columns if present, and also in meta_json).
    """
    model = str(model).strip()
    if model not in ("HW1F", "HW2F"):
        raise ValueError("model must be 'HW1F' or 'HW2F'.")

    if curve_snapshot is None:
        raise ValueError("curve_snapshot is required (run a calibration first, or ensure last_curve_snapshot is set).")

    created_at = _utc_now_iso()

    curve_json = json.dumps(curve_snapshot, ensure_ascii=False)
    params_json = json.dumps(params or {}, ensure_ascii=False)
    artifacts_json = json.dumps(artifacts or {}, ensure_ascii=False)
    notes_txt = "" if notes is None else str(notes)

    # merge meta
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
    Return a list of runs (latest first), already parsed from JSON.
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
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
    conn.commit()
    deleted = (cur.rowcount or 0) > 0
    conn.close()

    _clear_runs_cache()
    return deleted


def format_run_label(run: dict) -> str:
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
