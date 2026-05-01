"""CLI for producing class-balanced training datasets from features.csv.

Two strategies — pick whichever the modeling step expects:

    # 1) Random oversample minority class to match majority count.
    python -m scripts.balance_dataset oversample \
        --in data/processed/features.csv \
        --out data/processed/features_balanced.csv

    # 2) Bootstrap with Gaussian jitter (factor x dataset size).
    python -m scripts.balance_dataset bootstrap \
        --in data/processed/features.csv \
        --out data/processed/features_augmented.csv \
        --factor 8 --noise 0.02

Both strategies are deterministic given the seed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/balance_dataset.py …` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features.balance import bootstrap_augment, random_oversample


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class-balance the feature matrix.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    over = sub.add_parser("oversample",
                          help="Duplicate minority-class rows with replacement.")
    over.add_argument("--in", dest="in_csv", type=Path,
                      default=Path("data/processed/features.csv"))
    over.add_argument("--out", dest="out_csv", type=Path,
                      default=Path("data/processed/features_balanced.csv"))
    over.add_argument("--target", default="is_top_10")
    over.add_argument("--seed", type=int, default=42)

    boot = sub.add_parser("bootstrap",
                          help="Bootstrap with small Gaussian jitter on numeric features.")
    boot.add_argument("--in", dest="in_csv", type=Path,
                      default=Path("data/processed/features.csv"))
    boot.add_argument("--out", dest="out_csv", type=Path,
                      default=Path("data/processed/features_augmented.csv"))
    boot.add_argument("--target", default="is_top_10")
    boot.add_argument("--factor", type=int, default=8,
                      help="Output size multiplier vs input.")
    boot.add_argument("--noise", type=float, default=0.02,
                      help="Gaussian sigma as a fraction of each column's std-dev.")
    boot.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    df = pd.read_csv(args.in_csv)
    if args.cmd == "oversample":
        out = random_oversample(df, target_col=args.target, random_state=args.seed)
    else:
        out = bootstrap_augment(df, target_col=args.target,
                                factor=args.factor, noise=args.noise,
                                random_state=args.seed)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"{args.cmd}: {len(df)} → {len(out)} rows · "
          f"per-class {dict(out[args.target].value_counts())} · "
          f"→ {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
