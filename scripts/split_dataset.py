from __future__ import annotations

import json
import random
from pathlib import Path
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="./data/traces_export.jsonl")
    ap.add_argument("--outdir", default="./data/dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)
    in_path = Path(args.infile)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                continue

    random.shuffle(data)
    n = len(data)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    for name, ds in [("train", train), ("val", val), ("test", test)]:
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print({"counts": {"train": len(train), "val": len(val), "test": len(test)}, "out": str(out_dir)})


if __name__ == "__main__":
    main()


