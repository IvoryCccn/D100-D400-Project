from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CreditCardApproval.modeling import ID_COL, TARGET_COL, make_X_y
from CreditCardApproval.model_training import (
    split_data,
    train_glm,
    train_lgbm,
    get_categorical_indices,
)


def _toy_df_imbalanced(n: int = 300, pos_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)

    df = pd.DataFrame(
        {
            ID_COL: np.arange(1000, 1000 + n),
            "annual_income": rng.normal(200000, 40000, size=n).clip(50000, 600000),
            "children_number": rng.integers(0, 3, size=n),
            "family_size": rng.integers(1, 5, size=n),
            "age_years": rng.uniform(22, 70, size=n),
            "employed_years": rng.uniform(0, 30, size=n),

            "gender": rng.choice(["M", "F"], size=n),
            "own_car": rng.choice(["Y", "N"], size=n),
            "own_realty": rng.choice(["Y", "N"], size=n),
            "own_work_phone": rng.integers(0, 2, size=n),
            "own_phone": rng.integers(0, 2, size=n),
            "own_email": rng.integers(0, 2, size=n),

            "income_type": rng.choice(["Employed", "Retired", "Other"], size=n),
            "education_level": rng.choice(["Secondary", "Higher"], size=n),
            "family_status": rng.choice(["Married", "Single"], size=n),
            "housing_status": rng.choice(["Own_Home", "Rental", "With_Parents"], size=n),

            TARGET_COL: y,
        }
    )
    return df


def test_split_data_is_stratified():
    df = _toy_df_imbalanced(n=500, pos_rate=0.02, seed=42)
    X, y = make_X_y(df)

    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)

    train_rate = y_train.mean()
    val_rate = y_val.mean()
    assert abs(train_rate - val_rate) < 0.01


def test_get_categorical_indices_returns_positions():
    df = _toy_df_imbalanced(n=50, pos_rate=0.1, seed=7)
    X, _ = make_X_y(df)

    idx = get_categorical_indices(X)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    assert isinstance(idx, list)
    assert len(idx) == len(cat_cols)
    assert idx == [X.columns.get_loc(c) for c in cat_cols]
    
    
def test_train_glm_runs_and_returns_auc():
    df = _toy_df_imbalanced(n=400, pos_rate=0.03, seed=1)
    X, y = make_X_y(df)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)

    model, auc = train_glm(X_train, y_train, X_val, y_val)
    assert hasattr(model, "predict_proba")
    assert 0.0 <= auc <= 1.0


def test_train_glm_runs_with_sampling_method_smote_if_available():
    pytest.importorskip("imblearn")

    df = _toy_df_imbalanced(n=400, pos_rate=0.03, seed=11)
    X, y = make_X_y(df)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)

    model, auc = train_glm(X_train, y_train, X_val, y_val, sampling_method="smote")
    assert hasattr(model, "predict_proba")
    assert 0.0 <= auc <= 1.0
    

def test_train_lgbm_runs_and_returns_auc_if_installed():
    pytest.importorskip("lightgbm")

    df = _toy_df_imbalanced(n=400, pos_rate=0.03, seed=2)
    X, y = make_X_y(df)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)

    model, auc = train_lgbm(X_train, y_train, X_val, y_val)
    assert hasattr(model, "predict_proba")
    assert 0.0 <= auc <= 1.0


def test_train_lgbm_runs_with_sampling_method_undersample_if_installed():
    pytest.importorskip("lightgbm")
    pytest.importorskip("imblearn")

    df = _toy_df_imbalanced(n=400, pos_rate=0.03, seed=12)
    X, y = make_X_y(df)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)

    model, auc = train_lgbm(X_train, y_train, X_val, y_val, sampling_method="undersample")
    assert hasattr(model, "predict_proba")
    assert 0.0 <= auc <= 1.0
    