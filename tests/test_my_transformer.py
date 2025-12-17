import numpy as np
import pandas as pd
import pytest

from CreditCardApproval.feature_engineering.my_transformer import Log1pTransformer


# ==============================
# Fixtures
# ==============================

@pytest.fixture
def simple_df():
    """
    Simple DataFrame with non-negative numeric values.
    """
    return pd.DataFrame(
        {
            "annual_income": [0, 10000, 50000, 100000],
            "employed_years": [0, 1, 5, 10],
        }
    )


@pytest.fixture
def df_with_nan():
    """
    DataFrame containing NaN values.
    """
    return pd.DataFrame(
        {
            "annual_income": [0, np.nan, 50000],
        }
    )


@pytest.fixture
def df_with_negative():
    """
    DataFrame containing negative values (invalid for income).
    """
    return pd.DataFrame(
        {
            "annual_income": [10000, -500, 30000],
        }
    )


# ==============================
# Tests
# ==============================

def test_log1p_transformer_creates_new_column(simple_df):
    """
    Transformer should create a new log-transformed column.
    """
    transformer = Log1pTransformer(columns=["annual_income"])
    result = transformer.fit_transform(simple_df)

    assert "annual_income_log1p" in result.columns


def test_log1p_transformer_correct_values(simple_df):
    """
    Log1p transformation should be numerically correct.
    """
    transformer = Log1pTransformer(columns=["annual_income"])
    result = transformer.fit_transform(simple_df)

    expected = np.log1p(simple_df["annual_income"].values)
    actual = result["annual_income_log1p"].values

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_log1p_transformer_keeps_original_column(simple_df):
    """
    Original column should remain unchanged.
    """
    transformer = Log1pTransformer(columns=["annual_income"])
    result = transformer.fit_transform(simple_df)

    pd.testing.assert_series_equal(
        result["annual_income"],
        simple_df["annual_income"],
    )


def test_log1p_transformer_handles_nan(df_with_nan):
    """
    NaN values should remain NaN after log1p.
    """
    transformer = Log1pTransformer(columns=["annual_income"])
    result = transformer.fit_transform(df_with_nan)

    assert np.isnan(result.loc[1, "annual_income_log1p"])


def test_log1p_transformer_raises_on_negative(df_with_negative):
    """
    Negative values should raise a ValueError.
    """
    transformer = Log1pTransformer(columns=["annual_income"])

    with pytest.raises(ValueError):
        transformer.fit_transform(df_with_negative)


def test_log1p_transformer_missing_column_raises(simple_df):
    """
    Missing columns should raise a ValueError.
    """
    transformer = Log1pTransformer(columns=["non_existing_column"])

    with pytest.raises(ValueError):
        transformer.fit_transform(simple_df)


def test_log1p_transformer_multiple_columns(simple_df):
    """
    Transformer should support multiple columns.
    """
    transformer = Log1pTransformer(columns=["annual_income", "employed_years"])
    result = transformer.fit_transform(simple_df)

    assert "annual_income_log1p" in result.columns
    assert "employed_years_log1p" in result.columns
