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


@app.command()
def export_traces(out: str = "./data/traces_export.jsonl"):
    """Concatenate all trace_*.jsonl into one file for training."""
    cfg = load_config()
    traces_dir = Path(cfg.tracing.get("dir", "./data/traces"))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for p in sorted(traces_dir.glob("trace_*.jsonl")):
            try:
                with p.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        count += 1
            except Exception:
                continue
    typer.echo(json.dumps({"exported": count, "path": str(out_path)}))


if __name__ == "__main__":
    app()