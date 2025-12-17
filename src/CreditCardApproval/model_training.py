import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from CreditCardApproval.modeling import (
    FINAL_SCHEMA,
    make_glm_pipeline,
    make_lgbm_pipeline,
)


# ======================================================
# 1. Data spliting
# ======================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/validation split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ======================================================
# 2. Model training
# ======================================================

def train_glm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    """
    Train baseline logistic regression model and return AUC.
    """
    glm_pipe = make_glm_pipeline(
        schema=FINAL_SCHEMA,
        log_income=True,
        income_col="annual_income",
    )

    glm_pipe.fit(X_train, y_train)
    val_proba = glm_pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    return glm_pipe, auc


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    """
    Train baseline LightGBM model and return AUC.
    """
    lgbm_pipe = make_lgbm_pipeline(
        schema=FINAL_SCHEMA,
        log_income=True,
        income_col="annual_income",
    )

    lgbm_pipe.fit(X_train, y_train)
    val_proba = lgbm_pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    return lgbm_pipe, auc
