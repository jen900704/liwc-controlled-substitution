"""Strict CSV/config loading for the public command-line workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON configuration and resolve paths relative to its parent."""
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("input_csv", "output_dir"):
        if key not in config:
            raise ValueError(f"missing configuration key: {key}")
        candidate = Path(config[key])
        config[key] = candidate if candidate.is_absolute() else path.parent.parent / candidate
    return config


def load_feature_table(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load and validate the expected participant-level feature schema."""
    table = pd.read_csv(config["input_csv"])
    required = [config["dataset_column"], config["record_column"], config["label_column"]]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if table.duplicated(required[:2]).any():
        raise ValueError("dataset/record keys must be unique")
    base_columns = [c for c in table if c.startswith(config["base_prefix"])]
    liwc_columns = [c for c in table if c.startswith(config["liwc_prefix"])]
    if not base_columns or not liwc_columns:
        raise ValueError("at least one base and one LIWC feature are required")
    table[config["label_column"]] = pd.to_numeric(table[config["label_column"]], errors="raise").astype(int)
    if not set(table[config["label_column"]].unique()).issubset({0, 1}):
        raise ValueError("labels must be encoded as 0/1")
    return table, base_columns, liwc_columns
