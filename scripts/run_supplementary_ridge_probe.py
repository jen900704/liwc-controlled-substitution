#!/usr/bin/env python3
"""Supplementary single five-fold ridge probe; not the primary evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--base-prefix", default="base_")
    parser.add_argument("--liwc-prefix", default="liwc_")
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    base = [c for c in table if c.startswith(args.base_prefix)]
    liwc = [c for c in table if c.startswith(args.liwc_prefix)]
    if not base or not liwc:
        raise ValueError("base and LIWC columns are required")
    rows = []
    for dataset, part in table.groupby(args.dataset_column, sort=True):
        x, target = part[base].to_numpy(float), part[liwc].to_numpy(float)
        labels = part[args.label_column].to_numpy(int)
        _, counts = np.unique(labels, return_counts=True)
        splitter = (
            StratifiedKFold(5, shuffle=True, random_state=42)
            if len(counts) == 2 and counts.min() >= 5
            else KFold(5, shuffle=True, random_state=42)
        )
        prediction = np.zeros_like(target)
        split_labels = labels if isinstance(splitter, StratifiedKFold) else None
        for train, test in splitter.split(x, split_labels):
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
            model.fit(x[train], target[train])
            prediction[test] = model.predict(x[test])
        for column, feature in enumerate(liwc):
            rows.append({"dataset": dataset, "liwc_category": feature, "r2": r2_score(target[:, column], prediction[:, column])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} supplementary probe estimates to {args.output}")


if __name__ == "__main__":
    main()
