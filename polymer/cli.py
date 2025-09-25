from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from .config import load_config
from .orchestrator import Orchestrator
from .training.achmra import AchmraTrainingConfig, build_achmra_pipeline


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


@app.command()
def achmra(
    stage: str = typer.Option("status", help="One of: status, sft, preference, rl, evaluate, export"),
    config_path: str = typer.Option("config/training/achmra-base-solo.yaml", "--config", help="Path to ACHMRA training config"),
    split: str = typer.Option("val", help="Dataset split for evaluation"),
    checkpoint: str | None = typer.Option(None, help="Checkpoint directory when stage=export"),
    output: str | None = typer.Option(None, help="Optional GGUF export directory override"),
):
    """Manage the ACHMRA-Base-Solo training pipeline without auto-running training."""
    cfg = AchmraTrainingConfig.load(config_path)
    pipeline = build_achmra_pipeline(cfg)
    if stage == "status":
        payload = {
            "datasets": {name: len(ds) for name, ds in pipeline.datasets.items()},
            "export": cfg.export.model_dump(),
            "lora_enabled": cfg.lora.enabled,
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    if stage == "sft":
        pipeline.train_sft()
        return
    if stage == "preference":
        pipeline.train_preference()
        return
    if stage == "rl":
        pipeline.train_rl()
        return
    if stage == "evaluate":
        metrics = pipeline.evaluate(split)
        typer.echo(json.dumps({"split": split, "metrics": metrics}, indent=2))
        return
    if stage == "export":
        if not checkpoint:
            raise typer.BadParameter("--checkpoint is required when stage=export")
        outputs = pipeline.export_gguf(checkpoint, output)
        typer.echo(json.dumps({"exported": [str(p) for p in outputs]}, indent=2))
        return
    raise typer.BadParameter(f"Unknown stage {stage}")


if __name__ == "__main__":
    app()
