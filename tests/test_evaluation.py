from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from CreditCardApproval.evaluation import find_optimal_threshold, compare_models


def test_find_optimal_threshold_returns_valid_threshold_and_table():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])

    best_t, table = find_optimal_threshold(y_true=y_true, y_proba=y_proba, metric="f2")

    assert 0.0 <= best_t <= 1.0
    assert isinstance(table, pd.DataFrame)
    assert "threshold" in table.columns
    assert "f2" in table.columns


def test_compare_models_outputs_expected_columns():
    # directly employ LogisticRegression rather than pipelining
    X = pd.DataFrame({"x1": [0, 0, 1, 1, 2, 2], "x2": [0, 1, 0, 1, 0, 1]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    m = LogisticRegression(max_iter=1000).fit(X, y)

    summary = compare_models(models={"LR": m}, X=X, y=y, threshold=0.5)

    assert isinstance(summary, pd.DataFrame)

    for col in ["auc", "gini", "ks", "accuracy", "precision", "recall", "f1", "f2", "tp", "fp", "tn", "fn"]:
        assert col in summary.columns
    assert "LR" in summary.index
    