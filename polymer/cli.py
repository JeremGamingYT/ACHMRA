from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from .config import load_config
from .orchestrator import Orchestrator


app = typer.Typer(help="POLYMER agent CLI")


@app.command()
def serve(host: str | None = None, port: int | None = None):
    cfg = load_config()
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    uvicorn.run("polymer.server.app:app", host=cfg.server.host, port=cfg.server.port, reload=False)


@app.command()
def ingest(text: str):
    cfg = load_config()
    orc = Orchestrator(cfg)
    doc_id = orc.ingest(text)
    typer.echo(json.dumps({"doc_id": doc_id}))


@app.command()
def query(question: str):
    cfg = load_config()
    orc = Orchestrator(cfg)
    result = orc.query(question)
    typer.echo(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    app()