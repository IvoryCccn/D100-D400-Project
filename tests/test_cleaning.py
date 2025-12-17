from __future__ import annotations

import numpy as np
import pandas as pd

from CreditCardApproval.cleaning import (
    APPLICATION_RENAME_MAP,
    rename_application_columns,
    handle_application_missing,
    clip_outliers_application,
    recode_categorical_variables,
)


def test_rename_application_columns_does_not_mutate_original():
    df = pd.DataFrame(
        {
            "ID": [1],
            "AMT_INCOME_TOTAL": [100000.0],
            "CNT_CHILDREN": [0],
            "CNT_FAM_MEMBERS": [2],
        }
    )
    df_copy = df.copy(deep=True)

    out = rename_application_columns(df)

    # original unchanged
    pd.testing.assert_frame_equal(df, df_copy)

    # renamed columns exist
    for old, new in APPLICATION_RENAME_MAP.items():
        if old in df.columns:
            assert new in out.columns
            assert old not in out.columns


def test_handle_application_missing_sets_unemployed_employed_years_to_zero():
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "is_unemployed": [1, 0, 1],
            "employed_years": [np.nan, 5.0, 12.0],
        }
    )

    out = handle_application_missing(df)

    # unemployed -> employed_years = 0.0
    assert out.loc[out["ID"] == 1, "employed_years"].iloc[0] == 0.0
    assert out.loc[out["ID"] == 3, "employed_years"].iloc[0] == 0.0

    # employed -> keep value
    assert out.loc[out["ID"] == 2, "employed_years"].iloc[0] == 5.0


def test_clip_outliers_application_clips_by_quantiles():
    # make a numeric column with extreme outliers
    rng = np.random.default_rng(42)
    base_income = rng.normal(loc=200000, scale=20000, size=1000)
    income = np.concatenate([base_income, [10_000_000, 20_000_000]])  # extreme high
    employed_years = np.concatenate([rng.uniform(0, 20, size=1000), [50, 60]])

    df = pd.DataFrame(
        {
            "annual_income": income,
            "employed_years": employed_years,
            "gender": ["M"] * len(income),  # non-numeric
        }
    )

    lower_q, upper_q = 0.01, 0.99
    expected_income_lo, expected_income_hi = (
        df["annual_income"].quantile(lower_q),
        df["annual_income"].quantile(upper_q),
    )

    out = clip_outliers_application(df, lower_quantile=lower_q, upper_quantile=upper_q)

    # non-numeric should stay identical
    assert (out["gender"] == df["gender"]).all()

    # clipped bounds
    assert out["annual_income"].min() >= expected_income_lo - 1e-9
    assert out["annual_income"].max() <= expected_income_hi + 1e-9


def test_recode_categorical_variables_creates_expected_groups():
    df = pd.DataFrame(
        {
            "income_type": ["Working", "Pensioner", "Student", "Businessman"],
            "education_type": [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
            ],
            "family_status": ["Married", "Civil marriage", "Single / not married", "Widow"],
            "housing_type": ["House / apartment", "Rented apartment", "With parents", "Co-op apartment"],
        }
    )

    out = recode_categorical_variables(df)

    # Income type
    assert out["income_type_recoded"].tolist() == ["Employed", "Retired", "Other", "Self-Employed"]

    # Education level
    assert out["education_level"].tolist() == ["Secondary", "Higher", "Secondary", "Lower"]

    # Family status
    assert out["family_status_recoded"].tolist() == ["Married", "Married", "Single", "Previously_Married"]

    # Housing type
    assert out["housing_status"].tolist() == ["Own_Home", "Rental", "With_Parents", "Own_Home"]
