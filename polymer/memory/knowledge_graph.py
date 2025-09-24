from __future__ import annotations

from pathlib import Path
import networkx as nx


class KnowledgeGraph:
    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        if self.file_path.exists():
            try:
                self.graph = nx.read_graphml(self.file_path)
            except Exception:
                self.graph = nx.DiGraph()

    def add_fact(self, subject: str, predicate: str, object_: str) -> None:
        self.graph.add_node(subject)
        self.graph.add_node(object_)
        self.graph.add_edge(subject, object_, predicate=predicate)
        self._persist()

    def neighbors(self, node: str):
        return list(self.graph.successors(node))

    def _persist(self) -> None:
        try:
            nx.write_graphml(self.graph, self.file_path)
        except Exception:
            pass


