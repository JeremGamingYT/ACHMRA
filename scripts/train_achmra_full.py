from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from polymer.training.achmra import AchmraTrainingConfig, build_achmra_pipeline


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def _default_checkpoint(cfg: AchmraTrainingConfig, stage: str) -> Path:
    return Path(cfg.export.output_dir) / stage


def run_stage(label: str, func: Any) -> None:
    print(f"[{label}] start")
    func()
    print(f"[{label}] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end ACHMRA training driver")
    parser.add_argument("--config", default="config/training/achmra-base-solo.yaml", help="Training config path")
    parser.add_argument("--scenarios", help="Optional scenario yaml; presence will trigger dataset build guidance", default=None)
    parser.add_argument("--skip-sft", action="store_true", help="Skip the SFT stage")
    parser.add_argument("--run-preference", action="store_true", help="Run the preference stage after SFT")
    parser.add_argument("--run-rl", action="store_true", help="Run the RL stage after preference")
    parser.add_argument("--export", action="store_true", help="Export GGUF from the chosen checkpoint at the end")
    parser.add_argument("--checkpoint", help="Checkpoint directory for export (defaults to SFT output)")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned stages")
    args = parser.parse_args()

    cfg = AchmraTrainingConfig.load(args.config)

    required_paths = [cfg.data.train_file, cfg.data.val_file]
    if not args.skip_sft and cfg.data.train_file:
        required_paths.append(cfg.data.train_file)
    if args.run_preference and cfg.data.preference_file:
        required_paths.append(cfg.data.preference_file)
    if args.run_rl and cfg.data.rl_file:
        required_paths.append(cfg.data.rl_file)

    for file_path in required_paths:
        if file_path:
            _ensure_file(Path(file_path))

    planned = []
    if not args.skip_sft:
        planned.append("sft")
    if args.run_preference:
        planned.append("preference")
    if args.run_rl:
        planned.append("rl")
    if args.export:
        planned.append("export")

    print(json.dumps({"plan": planned, "config": args.config}, indent=2))
    if args.dry_run:
        return

    pipeline = build_achmra_pipeline(cfg)

    if not args.skip_sft:
        run_stage("sft", pipeline.train_sft)

    if args.run_preference:
        run_stage("preference", pipeline.train_preference)

    if args.run_rl:
        run_stage("rl", pipeline.train_rl)

    if args.export:
        checkpoint = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(cfg, "sft")
        outputs = pipeline.export_gguf(str(checkpoint))
        print(json.dumps({"exported": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
