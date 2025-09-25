from __future__ import annotations

import argparse
import json
from pathlib import Path

from polymer.training.achmra import AchmraTrainingConfig, build_achmra_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACHMRA training workflow stages")
    parser.add_argument("--config", default="/kaggle/working/config/training/achmra-base-solo.yaml", help="Path to the ACHMRA training config")
    parser.add_argument(
        "--stage",
        choices=["status", "sft", "preference", "rl", "evaluate", "export"],
        default="status",
        help="Pipeline stage to execute",
    )
    parser.add_argument("--split", default="val", help="Dataset split to evaluate when stage=evaluate")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory to export when stage=export (e.g. artifacts/achmra_base_solo/sft)",
    )
    parser.add_argument(
        "--output",
        help="Optional output directory override for GGUF export",
    )
    args = parser.parse_args()

    config = AchmraTrainingConfig.load(args.config)
    pipeline = build_achmra_pipeline(config)

    if args.stage == "status":
        payload = {
            "config": args.config,
            "datasets": {name: len(ds) for name, ds in pipeline.datasets.items()},
            "export": config.export.model_dump(),
            "lora_enabled": config.lora.enabled,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.stage == "sft":
        pipeline.train_sft()
        return

    if args.stage == "preference":
        pipeline.train_preference()
        return

    if args.stage == "rl":
        pipeline.train_rl()
        return

    if args.stage == "evaluate":
        metrics = pipeline.evaluate(args.split)
        print(json.dumps({"split": args.split, "metrics": metrics}, indent=2))
        return

    if args.stage == "export":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for export stage")
        outputs = pipeline.export_gguf(args.checkpoint, args.output)
        print(json.dumps({"exported": [str(p) for p in outputs]}, indent=2))
        return


if __name__ == "__main__":
    main()
