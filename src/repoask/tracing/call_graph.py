from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CallRelationship:
    caller: str
    callee: str
    file_path: str


class CallGraphStore:
    """SQLite-backed store for caller → callee relationships."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_graph (
                id TEXT PRIMARY KEY,
                caller TEXT NOT NULL,
                callee TEXT NOT NULL,
                file_path TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbol_locations (
                symbol TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_caller ON call_graph(caller)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_callee ON call_graph(callee)"
        )
        self._conn.commit()

    @staticmethod
    def _edge_id(caller: str, callee: str, file_path: str) -> str:
        return hashlib.sha256(f"{caller}|{callee}|{file_path}".encode()).hexdigest()

    def upsert(self, relationships: list[CallRelationship]) -> None:
        rows = [
            (
                self._edge_id(r.caller, r.callee, r.file_path),
                r.caller,
                r.callee,
                r.file_path,
            )
            for r in relationships
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO call_graph (id, caller, callee, file_path) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def upsert_locations(self, locations: list[tuple[str, str, int]]) -> None:
        """Store symbol → (file_path, line) mappings. Each tuple is (symbol, file_path, line)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO symbol_locations (symbol, file_path, line) VALUES (?, ?, ?)",
            locations,
        )
        self._conn.commit()

    def location_of(self, symbol: str) -> tuple[str, int] | None:
        """Return (file_path, line) for a symbol, or None if unknown."""
        row = self._conn.execute(
            "SELECT file_path, line FROM symbol_locations WHERE symbol = ?", (symbol,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def delete_by_file(self, file_path: str) -> None:
        self._conn.execute("DELETE FROM call_graph WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM symbol_locations WHERE file_path = ?", (file_path,))
        self._conn.commit()

    def callees_of(self, caller: str) -> list[str]:
        """Return all symbols directly called by `caller`."""
        rows = self._conn.execute(
            "SELECT callee FROM call_graph WHERE caller = ?", (caller,)
        ).fetchall()
        return [r[0] for r in rows]

    def callers_of(self, callee: str) -> list[str]:
        """Return all symbols that call `callee`."""
        rows = self._conn.execute(
            "SELECT caller FROM call_graph WHERE callee = ?", (callee,)
        ).fetchall()
        return [r[0] for r in rows]

    def all_callers(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT caller FROM call_graph"
        ).fetchall()
        return [r[0] for r in rows]

    def all_symbols(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT caller FROM call_graph UNION SELECT DISTINCT callee FROM call_graph"
        ).fetchall()
        return [r[0] for r in rows]

    def search_symbol(self, name: str) -> list[str]:
        """Fuzzy-ish search: return symbols containing `name` (case-insensitive)."""
        pattern = f"%{name.lower()}%"
        rows = self._conn.execute(
            """
            SELECT DISTINCT caller FROM call_graph WHERE LOWER(caller) LIKE ?
            UNION
            SELECT DISTINCT callee FROM call_graph WHERE LOWER(callee) LIKE ?
            """,
            (pattern, pattern),
        ).fetchall()
        return [r[0] for r in rows]

    def stats(self) -> dict:
        edges = self._conn.execute("SELECT COUNT(*) FROM call_graph").fetchone()[0]
        symbols = self._conn.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT DISTINCT caller FROM call_graph UNION SELECT DISTINCT callee FROM call_graph"
            ")"
        ).fetchone()[0]
        return {"edges": edges, "symbols": symbols}

    def close(self) -> None:
        self._conn.close()
