from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import List, Optional


class EpisodicMemory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)"
            )

    def add(self, event: str) -> int:
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("INSERT INTO episodes(event) VALUES (?)", (event,))
            con.commit()
            return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> List[str]:
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("SELECT event FROM episodes ORDER BY ts DESC LIMIT ?", (limit,))
            return [row[0] for row in cur.fetchall()]


