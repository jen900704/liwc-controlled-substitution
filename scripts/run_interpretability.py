#!/usr/bin/env python3
"""Compute category-level Cohen's d and deterministic participant bootstraps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liwc_controlled_substitution.statistics import participant_bootstrap_cohens_d, stable_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--liwc-prefix", default="liwc_")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    required = {args.dataset_column, args.label_column}
    if not required.issubset(table):
        raise ValueError(f"missing columns: {sorted(required - set(table.columns))}")
    features = [column for column in table if column.startswith(args.liwc_prefix)]
    if not features:
        raise ValueError("no LIWC feature columns found")
    rows = []
    for dataset, part in table.groupby(args.dataset_column, sort=True):
        for feature in features:
            seed = stable_seed("category", dataset, feature)
            result = participant_bootstrap_cohens_d(
                pd.to_numeric(part.loc[part[args.label_column] == 1, feature], errors="coerce"),
                pd.to_numeric(part.loc[part[args.label_column] == 0, feature], errors="coerce"),
                n_bootstrap=args.bootstrap, seed=seed,
            )
            rows.append({"dataset": dataset, "category": feature, **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} category estimates to {args.output}")


if __name__ == "__main__":
    main()
