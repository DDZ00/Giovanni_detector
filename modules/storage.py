"""SQLite persistence for the Giovanni research dashboard.

Phase 0: schema bootstrap + connection helper.
Phase 1: samples CRUD.
Phase 2: runs CRUD.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB_PATH = Path(os.environ.get("GIOVANNI_DB_PATH", "/app/data/giovanni.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    text                 TEXT NOT NULL,
    label                TEXT,
    english_proficiency  TEXT,
    source               TEXT,
    filename             TEXT,
    n_words              INTEGER NOT NULL,
    sha256               TEXT NOT NULL UNIQUE,
    notes                TEXT,
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);
-- idx_samples_english_proficiency is created by _migrate() so legacy DBs
-- (without the column yet) can still execute this script without error.

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id    INTEGER NOT NULL,
    params       TEXT NOT NULL,
    result       TEXT NOT NULL,
    duration_ms  INTEGER,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_sample_id ON runs(sample_id);

CREATE TABLE IF NOT EXISTS presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    params      TEXT NOT NULL,
    notes       TEXT,
    created_at  TEXT NOT NULL
);
"""

VALID_LABELS = {
    "human", "synthetic", "unknown",
}

VALID_PROFICIENCY = {
    "native", "non_native", "unknown",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _configure(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    try:
        _configure(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
    return path


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent migrations for DBs created before a schema change.

    SQLite's ALTER TABLE ADD COLUMN is the safest forward migration: it sets
    NULL on every existing row and is a no-op if the column already exists
    (we detect that via PRAGMA table_info).

    The english_proficiency index lives here (not in SCHEMA) so that legacy
    DBs without the column can still run the SCHEMA script without errors —
    SCHEMA stays a "fresh-DB blueprint" and _migrate() owns column-dependent
    artefacts.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(samples)")}
    if "english_proficiency" not in cols:
        conn.execute("ALTER TABLE samples ADD COLUMN english_proficiency TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_english_proficiency "
                 "ON samples(english_proficiency)")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ── samples CRUD ───────────────────────────────────────────────
class DuplicateSample(Exception):
    pass


def add_sample(
    text: str,
    *,
    label: Optional[str] = None,
    english_proficiency: Optional[str] = None,
    source: Optional[str] = None,
    filename: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    if not text.strip():
        raise ValueError("text is empty")
    if label is not None and label not in VALID_LABELS:
        raise ValueError(f"invalid label {label!r}; allowed: {sorted(VALID_LABELS)}")
    if english_proficiency is not None and english_proficiency not in VALID_PROFICIENCY:
        raise ValueError(
            f"invalid english_proficiency {english_proficiency!r}; "
            f"allowed: {sorted(VALID_PROFICIENCY)}"
        )
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    n_words = len(text.split())
    created_at = _now()
    with connect(db_path) as conn:
        try:
            cur = conn.execute(
                "INSERT INTO samples(text, label, english_proficiency, source, filename, "
                "n_words, sha256, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (text, label, english_proficiency, source, filename,
                 n_words, sha, notes, created_at),
            )
        except sqlite3.IntegrityError as e:
            if "samples.sha256" in str(e):
                existing = conn.execute("SELECT id FROM samples WHERE sha256 = ?", (sha,)).fetchone()
                raise DuplicateSample(f"sample already exists with id={existing['id']}")
            raise
        return {
            "id": cur.lastrowid,
            "label": label, "english_proficiency": english_proficiency,
            "source": source, "filename": filename,
            "n_words": n_words, "sha256": sha, "notes": notes,
            "created_at": created_at,
        }


def list_samples(
    *, label: Optional[str] = None, english_proficiency: Optional[str] = None,
    limit: Optional[int] = None, offset: int = 0,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    sql = ("SELECT id, label, english_proficiency, source, filename, n_words, "
           "sha256, notes, created_at FROM samples")
    where: list = []
    args: list = []
    if label is not None:
        where.append("label = ?")
        args.append(label)
    if english_proficiency is not None:
        where.append("english_proficiency = ?")
        args.append(english_proficiency)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        args.extend([limit, offset])
    with connect(db_path) as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, args).fetchall()]


def get_sample(sample_id: int, *, db_path: Path | str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
        return _row_to_dict(row) if row else None


def delete_sample(sample_id: int, *, db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    with connect(db_path) as conn:
        cur = conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
        return cur.rowcount > 0


def export_samples_jsonl(*, db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[str]:
    with connect(db_path) as conn:
        for row in conn.execute("SELECT * FROM samples ORDER BY id ASC"):
            yield json.dumps(_row_to_dict(row), ensure_ascii=False) + "\n"


# ── runs CRUD ──────────────────────────────────────────────────
def add_run(
    sample_id: int, params: Dict[str, Any], result: Dict[str, Any],
    duration_ms: Optional[int] = None,
    *, db_path: Path | str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    created_at = _now()
    with connect(db_path) as conn:
        if not conn.execute("SELECT 1 FROM samples WHERE id = ?", (sample_id,)).fetchone():
            raise ValueError(f"no sample with id={sample_id}")
        cur = conn.execute(
            "INSERT INTO runs(sample_id, params, result, duration_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (sample_id, json.dumps(params, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False, default=_json_default),
             duration_ms, created_at),
        )
        return {"id": cur.lastrowid, "sample_id": sample_id, "duration_ms": duration_ms, "created_at": created_at}


def list_runs(sample_id: int, *, db_path: Path | str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, sample_id, params, duration_ms, created_at FROM runs "
            "WHERE sample_id = ? ORDER BY id DESC", (sample_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = _row_to_dict(r)
            d["params"] = json.loads(d["params"])
            out.append(d)
        return out


def get_run(run_id: int, *, db_path: Path | str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        d["params"] = json.loads(d["params"])
        d["result"] = json.loads(d["result"])
        return d


def delete_run(run_id: int, *, db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    with connect(db_path) as conn:
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


def export_runs_jsonl(*, db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[str]:
    with connect(db_path) as conn:
        for row in conn.execute("SELECT * FROM runs ORDER BY id ASC"):
            d = _row_to_dict(row)
            d["params"] = json.loads(d["params"])
            d["result"] = json.loads(d["result"])
            yield json.dumps(d, ensure_ascii=False) + "\n"


def latest_run_per_sample(*, db_path: Path | str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """For corpus scatter views: each sample's most recent run with label."""
    with connect(db_path) as conn:
        rows = conn.execute("""
            SELECT s.id AS sample_id, s.label, s.n_words, r.id AS run_id, r.result, r.created_at
            FROM samples s
            JOIN runs r ON r.id = (
                SELECT id FROM runs WHERE sample_id = s.id ORDER BY id DESC LIMIT 1
            )
            ORDER BY s.id DESC
        """).fetchall()
        out = []
        for r in rows:
            d = _row_to_dict(r)
            d["result"] = json.loads(d["result"])
            out.append(d)
        return out


def counts(*, db_path: Path | str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """Resilient count for /health: returns zeros if the DB is unavailable
    (uninitialised host, test runs outside Docker, etc.) instead of 500-ing."""
    if not Path(db_path).exists():
        return {"samples": 0, "runs": 0}
    try:
        with connect(db_path) as conn:
            n_samples = conn.execute("SELECT COUNT(*) AS c FROM samples").fetchone()["c"]
            n_runs = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
            return {"samples": n_samples, "runs": n_runs}
    except sqlite3.Error:
        return {"samples": 0, "runs": 0}


def _json_default(obj: Any) -> Any:
    """Coerce numpy scalars / arrays so analyze_text() outputs serialise cleanly."""
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(obj)}")
