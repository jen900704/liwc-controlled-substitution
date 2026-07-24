"""Matched repeated-CV controlled-substitution evaluation."""

from __future__ import annotations

import copy
import json
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .controls import make_liwc_control


def experiment_seed(*items: object) -> int:
    """Reproduce the CRC32 seed construction used by the primary runner."""
    raw = "|".join(map(str, items)).encode("utf-8")
    return 100000 + zlib.crc32(raw) % 900000


def default_models() -> dict[str, object]:
    """Return the two validated linear classifiers."""
    return {
        "logreg": LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced",
            max_iter=5000, random_state=42,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced", random_state=42, max_iter=10000,
        ),
    }


@dataclass(frozen=True)
class EvaluationConfig:
    """Primary evaluation settings."""

    n_splits: int = 5
    n_repeats: int = 10
    random_state: int = 42
    control_draws: int = 30


def _fit_evaluate(model: object, train: np.ndarray, test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", copy.deepcopy(model)),
    ])
    pipeline.fit(train, y_train)
    prediction = pipeline.predict(test)
    try:
        auroc = roc_auc_score(y_test, pipeline.decision_function(test))
    except (AttributeError, ValueError):
        auroc = float("nan")
    return {
        "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
        "recall_pos": recall_score(y_test, prediction, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "auroc": auroc,
    }


def evaluate_dataset(
    dataset: str,
    base: np.ndarray,
    liwc: np.ndarray,
    labels: np.ndarray,
    *,
    config: EvaluationConfig = EvaluationConfig(),
    models: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Evaluate all matched conditions on identical participant-level splits."""
    base, liwc, labels = np.asarray(base, float), np.asarray(liwc, float), np.asarray(labels, int)
    if base.ndim != 2 or liwc.ndim != 2 or len(base) != len(liwc) or len(base) != len(labels):
        raise ValueError("base, LIWC, and labels must align row-for-row")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("binary labels are required")
    n_splits = min(config.n_splits, int(counts.min()))
    if n_splits < 2:
        raise ValueError("each class needs at least two records")
    if config.control_draws < 1:
        raise ValueError("control_draws must be positive")
    estimators = models or default_models()
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=config.n_repeats, random_state=config.random_state
    )
    rows: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(base, labels), 1):
        y_train, y_test = labels[train_index], labels[test_index]
        base_train, base_test = base[train_index], base[test_index]
        liwc_train, liwc_test = liwc[train_index], liwc[test_index]
        for variant in ("base", "intact", "pca", "shuffled", "random"):
            draws = range(config.control_draws) if variant in {"shuffled", "random"} else range(1)
            for draw in draws:
                if variant == "base":
                    x_train, x_test = base_train, base_test
                else:
                    rng = np.random.default_rng(experiment_seed(dataset, "route", variant, fold, draw))
                    control_train, control_test = make_liwc_control(variant, liwc_train, liwc_test, rng)
                    x_train = np.concatenate([base_train, control_train], axis=1)
                    x_test = np.concatenate([base_test, control_test], axis=1)
                for model_name, model in estimators.items():
                    rows.append({
                        "dataset": dataset, "route": "route", "variant": variant,
                        "model": model_name, "fold": fold, "repeat": (fold - 1) // n_splits + 1,
                        "outer_fold": (fold - 1) % n_splits + 1, "control_draw": draw,
                        "n_total": len(labels), "n_train": len(train_index), "n_test": len(test_index),
                        "class_counts": json.dumps(dict(zip(map(int, classes), map(int, counts)))),
                        **_fit_evaluate(model, x_train, x_test, y_train, y_test),
                    })
    return pd.DataFrame(rows)


def average_control_draws(raw: pd.DataFrame) -> pd.DataFrame:
    """Average shuffled/random draws within each matched fold."""
    keys = ["dataset", "route", "variant", "model", "fold", "repeat", "outer_fold", "n_total", "n_train", "n_test", "class_counts"]
    metrics = ["macro_f1", "recall_pos", "balanced_accuracy", "auroc"]
    return raw.groupby(keys, as_index=False)[metrics].mean()


def summarize_conditions(averaged: pd.DataFrame) -> pd.DataFrame:
    """Average fold scores to one dataset-route-model-condition row."""
    metrics = ["macro_f1", "recall_pos", "balanced_accuracy", "auroc"]
    aggregations = {metric: [("mean", "mean"), ("std", "std")] for metric in metrics}
    named = {f"{metric}_{suffix}": pd.NamedAgg(column=metric, aggfunc=func)
             for metric, pairs in aggregations.items() for suffix, func in pairs}
    return averaged.groupby(["dataset", "route", "variant", "model"], as_index=False).agg(**named)


def dataset_level_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    """Average model-matched intact-minus-comparator macro-F1 within dataset."""
    wide = summary.pivot(index=["dataset", "route", "model"], columns="variant", values="macro_f1_mean")
    required = {"intact", "base", "pca", "shuffled", "random"}
    if not required.issubset(wide.columns):
        raise ValueError(f"missing conditions: {sorted(required - set(wide.columns))}")
    rows = []
    for comparator in ("base", "pca", "shuffled", "random"):
        delta = (wide["intact"] - wide[comparator]).rename("delta").reset_index()
        blocked = delta.groupby(["dataset", "route"], as_index=False)["delta"].mean()
        blocked["contrast"] = f"intact-{comparator}"
        rows.append(blocked)
    return pd.concat(rows, ignore_index=True)
