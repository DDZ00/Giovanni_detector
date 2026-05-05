"""Tests for the english_proficiency column and its idempotent migration.

Uses a tempfile-backed SQLite so we can reset the schema between tests.
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from modules import storage


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "giovanni.db"


def _create_old_schema(path: Path) -> None:
    """Mimics the pre-migration schema (no english_proficiency column)."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript("""
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label TEXT,
                source TEXT,
                filename TEXT,
                n_words INTEGER NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                notes TEXT,
                created_at TEXT NOT NULL
            );
            INSERT INTO samples(text, label, source, filename, n_words, sha256, notes, created_at)
            VALUES ('legacy text body here', 'human', 'Old Corpus', 'old.txt', 4,
                    'legacysha256', NULL, '2026-04-01T00:00:00+00:00');
        """)
        conn.commit()
    finally:
        conn.close()


def _columns(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    finally:
        conn.close()


class TestMigration:
    def test_fresh_db_has_new_column(self, db_path):
        storage.init_db(db_path)
        assert "english_proficiency" in _columns(db_path)

    def test_old_db_gets_column_added(self, db_path):
        _create_old_schema(db_path)
        assert "english_proficiency" not in _columns(db_path)
        storage.init_db(db_path)
        assert "english_proficiency" in _columns(db_path)

    def test_old_rows_get_null_after_migration(self, db_path):
        _create_old_schema(db_path)
        storage.init_db(db_path)
        rows = storage.list_samples(db_path=db_path)
        assert len(rows) == 1
        assert rows[0]["english_proficiency"] is None
        # Existing data is preserved.
        assert rows[0]["label"] == "human"
        assert rows[0]["source"] == "Old Corpus"

    def test_init_db_is_idempotent(self, db_path):
        storage.init_db(db_path)
        storage.init_db(db_path)  # second call must not fail
        cols = _columns(db_path)
        # No duplicate columns; the new one is still present exactly once.
        assert sum(1 for c in cols if c == "english_proficiency") == 1


class TestProficiencyValidation:
    def test_valid_proficiency_round_trips(self, db_path):
        storage.init_db(db_path)
        rec = storage.add_sample(
            "some sample text body that is non empty",
            label="human", english_proficiency="non_native",
            db_path=db_path,
        )
        assert rec["english_proficiency"] == "non_native"
        fetched = storage.get_sample(rec["id"], db_path=db_path)
        assert fetched["english_proficiency"] == "non_native"

    def test_invalid_proficiency_rejected(self, db_path):
        storage.init_db(db_path)
        with pytest.raises(ValueError, match="invalid english_proficiency"):
            storage.add_sample(
                "text", label="human", english_proficiency="not-a-real-value",
                db_path=db_path,
            )

    def test_filter_by_proficiency(self, db_path):
        storage.init_db(db_path)
        storage.add_sample("text one alpha", english_proficiency="native", db_path=db_path)
        storage.add_sample("text two beta", english_proficiency="non_native", db_path=db_path)
        storage.add_sample("text three gamma", english_proficiency=None, db_path=db_path)
        natives = storage.list_samples(english_proficiency="native", db_path=db_path)
        assert len(natives) == 1
        assert natives[0]["english_proficiency"] == "native"
