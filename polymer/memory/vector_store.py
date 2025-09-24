from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import sqlite3
import numpy as np
from fastembed import TextEmbedding


class VectorStore:
    def __init__(self, db_path: Path, embed_model: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = TextEmbedding(model_name=embed_model)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, metadata TEXT)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS vectors (doc_id INTEGER, dim INTEGER, data BLOB, FOREIGN KEY(doc_id) REFERENCES docs(id))"
            )

    def add_text(self, text: str, metadata: dict | None = None) -> int:
        import json

        vector = self._embed([text])[0]
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("INSERT INTO docs(text, metadata) VALUES (?, ?)", (text, json.dumps(metadata or {})))
            doc_id = cur.lastrowid
            blob = vector.astype(np.float32).tobytes()
            cur.execute("INSERT INTO vectors(doc_id, dim, data) VALUES (?, ?, ?)", (doc_id, len(vector), blob))
            con.commit()
        return int(doc_id)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        q = self._embed([query])[0]
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("SELECT doc_id, dim, data FROM vectors")
            results: List[Tuple[int, float, str]] = []
            for doc_id, dim, data in cur.fetchall():
                vec = np.frombuffer(data, dtype=np.float32)
                sim = self._cosine_similarity(q, vec)
                results.append((doc_id, float(sim), ""))
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            # fetch texts
            enriched: List[Tuple[int, float, str]] = []
            for doc_id, score, _ in results:
                cur.execute("SELECT text FROM docs WHERE id = ?", (doc_id,))
                row = cur.fetchone()
                enriched.append((int(doc_id), float(score), row[0] if row else ""))
            return enriched

    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        vectors = list(self.embedder.embed(texts))
        return [np.array(v, dtype=np.float32) for v in vectors]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-6
        return float(np.dot(a, b) / denom)


