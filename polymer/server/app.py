from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..config import load_config
from ..orchestrator import Orchestrator


class IngestReq(BaseModel):
    text: str
    metadata: dict | None = None


class QueryReq(BaseModel):
    question: str


cfg = load_config()
orc = Orchestrator(cfg)
app = FastAPI(title="POLYMER Agent")


@app.get("/status")
def status():
    return {"llm": cfg.llm.model or cfg.llm.provider, "memory": cfg.memory.model_dump(), "planner": cfg.planner.model_dump()}


@app.post("/ingest")
def ingest(req: IngestReq):
    doc_id = orc.ingest(req.text, req.metadata)
    return {"doc_id": doc_id}


@app.post("/query")
def query(req: QueryReq):
    return orc.query(req.question)


