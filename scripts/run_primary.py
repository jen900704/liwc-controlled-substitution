#!/usr/bin/env python3
"""Run the primary repeated-CV substitution analysis from a JSON config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from liwc_controlled_substitution.evaluation import (
    EvaluationConfig, average_control_draws, dataset_level_contrasts,
    default_models, evaluate_dataset, summarize_conditions,
)
from liwc_controlled_substitution.io import load_config, load_feature_table
from liwc_controlled_substitution.statistics import benjamini_hochberg, exact_sign_flip_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    table, base_columns, liwc_columns = load_feature_table(config)
    selected = set(config.get("models", ["logreg", "linear_svm"]))
    models = {name: model for name, model in default_models().items() if name in selected}
    if not models:
        raise ValueError("no supported model selected")
    evaluation = EvaluationConfig(
        n_splits=int(config.get("n_splits", 5)),
        n_repeats=int(config.get("n_repeats", 10)),
        random_state=int(config.get("random_state", 42)),
        control_draws=int(config.get("control_draws", 30)),
    )
    outputs = []
    dataset_column, label_column = config["dataset_column"], config["label_column"]
    for dataset, part in table.groupby(dataset_column, sort=True):
        outputs.append(evaluate_dataset(
            str(dataset), part[base_columns].to_numpy(float), part[liwc_columns].to_numpy(float),
            part[label_column].to_numpy(int), config=evaluation, models=models,
        ))
    raw = pd.concat(outputs, ignore_index=True)
    averaged = average_control_draws(raw)
    summary = summarize_conditions(averaged)
    contrasts = dataset_level_contrasts(summary)
    tests = contrasts.groupby(["route", "contrast"])["delta"].agg(list).reset_index()
    tests["n_datasets"] = tests["delta"].map(len)
    tests["mean_delta"] = tests["delta"].map(lambda x: float(pd.Series(x).mean()))
    tests["p"] = tests["delta"].map(exact_sign_flip_test)
    tests["q"] = benjamini_hochberg(tests["p"])
    tests = tests.drop(columns="delta")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "fold_results_raw.csv", index=False)
    averaged.to_csv(output / "fold_results_averaged_controls.csv", index=False)
    summary.to_csv(output / "condition_summary.csv", index=False)
    contrasts.to_csv(output / "dataset_level_contrasts.csv", index=False)
    tests.to_csv(output / "exact_tests_bh.csv", index=False)
    print(f"Wrote {len(raw)} raw evaluations and {len(tests)} tests to {output}")


if __name__ == "__main__":
    main()
