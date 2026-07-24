"""Fold-local LIWC control constructions."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

Control = Literal["intact", "pca", "shuffled", "random"]


def make_liwc_control(
    control: Control,
    train: ArrayLike,
    test: ArrayLike,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct a LIWC condition using training-fold information only.

    ``pca`` median-imputes, standardizes, fits PCA on the training fold, and
    zero-pads to the original width when the fold permits fewer components.
    ``shuffled`` permutes complete training rows and samples complete training
    rows for test cases. ``random`` samples every feature independently from
    its imputed training-fold marginal. The intact condition returns the
    original arrays so the downstream fold-local pipeline performs imputation.
    """
    x_train = np.asarray(train, dtype=float)
    x_test = np.asarray(test, dtype=float)
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("train and test must be two-dimensional")
    if x_train.shape[1] != x_test.shape[1] or x_train.shape[1] == 0:
        raise ValueError("train and test must have the same nonzero feature width")
    if x_train.shape[0] < 2:
        raise ValueError("at least two training rows are required")
    if control == "intact":
        return x_train.copy(), x_test.copy()

    imputer = SimpleImputer(strategy="median")
    train_filled = imputer.fit_transform(x_train)
    test_filled = imputer.transform(x_test)
    n_train, n_features = train_filled.shape
    n_test = test_filled.shape[0]

    if control == "pca":
        components = min(n_features, max(1, n_train - 1))
        scaler = StandardScaler()
        z_train = scaler.fit_transform(train_filled)
        z_test = scaler.transform(test_filled)
        pca = PCA(n_components=components, random_state=0)
        out_train = pca.fit_transform(z_train)
        out_test = pca.transform(z_test)
        if components < n_features:
            padding = ((0, 0), (0, n_features - components))
            out_train = np.pad(out_train, padding)
            out_test = np.pad(out_test, padding)
        return out_train, out_test

    if control == "shuffled":
        return (
            train_filled[rng.permutation(n_train)],
            train_filled[rng.integers(0, n_train, size=n_test)],
        )

    if control == "random":
        out_train = np.empty_like(train_filled)
        out_test = np.empty_like(test_filled)
        for column in range(n_features):
            marginal = train_filled[:, column]
            out_train[:, column] = rng.choice(marginal, size=n_train, replace=True)
            out_test[:, column] = rng.choice(marginal, size=n_test, replace=True)
        return out_train, out_test

    raise ValueError(f"unknown LIWC control: {control}")
