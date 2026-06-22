from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class FileTracker:
    """Tracks file content hashes to enable incremental indexing."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS file_hashes "
            "(path TEXT PRIMARY KEY, hash TEXT NOT NULL, indexed_at REAL NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def is_changed(self, path: Path, root: Path) -> bool:
        rel = str(path.relative_to(root))
        current_hash = self.hash_file(path)
        row = self._conn.execute(
            "SELECT hash FROM file_hashes WHERE path = ?", (rel,)
        ).fetchone()
        return row is None or row[0] != current_hash

    def mark_indexed(self, path: Path, root: Path) -> None:
        import time
        rel = str(path.relative_to(root))
        current_hash = self.hash_file(path)
        self._conn.execute(
            "INSERT OR REPLACE INTO file_hashes (path, hash, indexed_at) VALUES (?, ?, ?)",
            (rel, current_hash, time.time()),
        )
        self._conn.commit()

    def remove(self, path: Path, root: Path) -> None:
        rel = str(path.relative_to(root))
        self._conn.execute("DELETE FROM file_hashes WHERE path = ?", (rel,))
        self._conn.commit()

    def all_indexed_paths(self) -> list[str]:
        rows = self._conn.execute("SELECT path FROM file_hashes").fetchall()
        return [r[0] for r in rows]

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
