from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CreditCardApproval.modeling import (
    FINAL_SCHEMA,
    ID_COL,
    TARGET_COL,
    validate_schema,
    make_X_y,
    make_glm_pipeline,
    make_lgbm_pipeline,
)


def _toy_processed_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            ID_COL: np.arange(1000, 1000 + n),
            "annual_income": rng.normal(200000, 30000, size=n).clip(50000, 500000),
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

            TARGET_COL: rng.integers(0, 2, size=n),
        }
    )
    return df


def test_validate_schema_passes_on_valid_df():
    df = _toy_processed_df()
    validate_schema(df, FINAL_SCHEMA)  # should not raise


def test_validate_schema_raises_when_missing_columns():
    df = _toy_processed_df().drop(columns=["annual_income"])
    with pytest.raises(ValueError):
        validate_schema(df, FINAL_SCHEMA)


def test_make_X_y_splits_and_drops_id():
    df = _toy_processed_df()
    X, y = make_X_y(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert TARGET_COL not in X.columns
    assert ID_COL not in X.columns
    assert len(X) == len(y) == len(df)


def test_glm_pipeline_can_fit_and_predict_proba():
    df = _toy_processed_df(n=80, seed=1)
    X, y = make_X_y(df)

    pipe = make_glm_pipeline(schema=FINAL_SCHEMA, log_income=True, income_col="annual_income")
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_lgbm_pipeline_can_fit_and_predict_proba_if_installed():
    pytest.importorskip("lightgbm")  # 如果没装 lightgbm，这个测试会自动跳过

    df = _toy_processed_df(n=80, seed=2)
    X, y = make_X_y(df)

    pipe = make_lgbm_pipeline(schema=FINAL_SCHEMA, log_income=True, income_col="annual_income")
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))
