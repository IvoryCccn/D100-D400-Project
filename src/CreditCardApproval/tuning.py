from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from CreditCardApproval.modeling import (
    FINAL_SCHEMA,
    make_glm_pipeline,
    make_lgbm_pipeline,
)


# ======================================================
# 1. GLM tuning
# ======================================================

def tune_glm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune logistic regression (GLM) hyperparameters using GridSearchCV.
    """
    glm_pipe = make_glm_pipeline(
        schema=FINAL_SCHEMA,
        log_income=True,
        income_col="annual_income",
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
    }

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=glm_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_


# ======================================================
# 2. LGBM tuning
# ======================================================

def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune LightGBM hyperparameters using GridSearchCV.
    """
    lgbm_pipe = make_lgbm_pipeline(
        schema=FINAL_SCHEMA,
        log_income=True,
        income_col="annual_income",
    )

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__num_leaves": [15, 31, 63],
        "model__learning_rate": [0.03, 0.05, 0.1],
    }

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=lgbm_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_
