from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from CreditCardApproval.feature_engineering import Log1pTransformer

from typing import Any, Dict


# ======================================================
# 1. Configuration & Feature Schema
# ======================================================

ID_COL: str = "ID"
TARGET_COL: str = "target"


@dataclass(frozen=True)
class FeatureSchema:
    """
    A simple container for feature groups used by sklearn ColumnTransformer.
    """
    numeric: List[str]
    categorical: List[str]

    def all_features(self) -> List[str]:
        return self.numeric + self.categorical


FINAL_SCHEMA = FeatureSchema(
    numeric=[
        "annual_income",
        "children_number",
        "family_size",
        "age_years",
        "employed_years",
        "own_work_phone",
        "own_phone",
        "own_email",
    ],
    categorical=[
        "gender",
        "own_car",
        "own_realty",
        "income_type",
        "education_level",
        "family_status",
        "housing_status",
    ],
)


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
) -> None:
    """
    Validate that all required columns exist in df.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_schema(
    df: pd.DataFrame,
    schema: FeatureSchema,
    id_col: str = ID_COL,
    target_col: str = TARGET_COL,
) -> None:
    """
    Validate modelling schema:
        - id_col and target_col exist
        - all features exist
        - no overlap between feature groups
    """
    # two main features existence check
    validate_required_columns(df, [id_col, target_col])

    # feature existence check
    validate_required_columns(df, schema.all_features())

    # group overlap check
    overlap = set(schema.numeric).intersection(schema.categorical)
    if overlap:
        raise ValueError(f"Overlapping features in numeric & categorical groups: {sorted(overlap)}")


# ======================================================
# 2. Data Interface (X/y Builder)
# ======================================================

def make_X_y(
    df: pd.DataFrame,
    id_col: str = ID_COL,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target vector y from an applicant-level dataframe.
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    return X, y


# ======================================================
# 3. Preprocessing Pipelines (ColumnTransformer)
# ======================================================

def make_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for numeric & categorical preprocessing.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, schema.numeric),
            ("cat", categorical_pipe, schema.categorical),
        ],
        remainder="drop",
    )

    return preprocessor


# ======================================================
# 4. Model Pipelines (GLM & LGBM)
# ======================================================

def make_glm_pipeline(
    schema: FeatureSchema,
    log_income: bool = True,
    income_col: str = "annual_income",
    random_state: int = 42,
) -> Pipeline:
    """
    Build a GLM pipeline: feature engineering + preprocessing + model.
    """
    steps = []

    if log_income and income_col in (schema.numeric + schema.categorical):
        # create new log feature while keep original numeric list
        steps.append(("log_income", Log1pTransformer(columns=[income_col], add_suffix=True)))

        # include the new log column into numeric features
        schema = FeatureSchema(
            numeric=schema.numeric + [f"{income_col}_log1p"],
            categorical=schema.categorical,
        )

    preprocessor = make_preprocessor(schema)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=None,
    )

    steps.extend(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return Pipeline(steps=steps)


def make_lgbm_pipeline(
    schema: FeatureSchema,
    log_income: bool = True,
    income_col: str = "annual_income",
    random_state: int = 42,
):
    """
    Build an LGBM pipeline: feature engineering + preprocessing + model.
    """
    from lightgbm import LGBMClassifier

    steps = []

    if log_income and income_col in (schema.numeric + schema.categorical):
        steps.append(("log_income", Log1pTransformer(columns=[income_col], add_suffix=True)))
        schema = FeatureSchema(
            numeric=schema.numeric + [f"{income_col}_log1p"],
            categorical=schema.categorical,
        )

    preprocessor = make_preprocessor(schema)

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
    )

    steps.extend(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return Pipeline(steps=steps)

"""
A waning might happened saying "X does not have valid feature names, but 
LGBMClassifier was fitted with feature names", which is not important, so
I simply ignore it in order to perform a clean pipeline outcome in notebook.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


# ======================================================
# 5. Others: Default Params & Sanity Checks
# ======================================================

def summarize_target(y: pd.Series) -> pd.DataFrame:
    """
    Return a small summary table for the binary target distribution.
    """
    counts = y.value_counts(dropna=False).sort_index()
    props = (counts / counts.sum()).rename("proportion")
    out = pd.concat([counts.rename("count"), props], axis=1)
    return out


def preview_pipeline_dimensions(
    pipeline,
    X: pd.DataFrame,
    n_rows: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Preview the transformed feature dimension after preprocessing, to sanity check one-hot expansion without training the estimator.
    """
    if "preprocess" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain a step named 'preprocess'.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    X_sample = (
        X.sample(n=min(n_rows, len(X)), random_state=random_state)
        if len(X) > n_rows
        else X
    )

    # 1) run all steps BEFORE preprocess (e.g., log_income) to create derived columns
    Xt_df = X_sample.copy()
    for name, step in pipeline.steps:
        if name == "preprocess":
            break
        Xt_df = step.fit_transform(Xt_df)

    # 2) now run preprocess only
    preprocess = pipeline.named_steps["preprocess"]
    Xt = preprocess.fit_transform(Xt_df)

    return {
        "n_features_after_preprocess": Xt.shape[1],
        "n_samples_preview": Xt.shape[0],
        "preview_rows_used": len(X_sample),
        "columns_after_fe_preview": list(Xt_df.columns),
    }