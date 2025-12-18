from typing import Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from CreditCardApproval.modeling import (
    FINAL_SCHEMA,
    make_glm_pipeline,
    make_lgbm_pipeline,
)

from sklearn.metrics import roc_auc_score


# ======================================================
# Sampling
# ======================================================

def get_categorical_indices(X: pd.DataFrame) -> list:
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    return [X.columns.get_loc(col) for col in categorical_cols]


def apply_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_method: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply sampling method to training data.
    """
    if sampling_method is None or sampling_method == "none":
        return X_train, y_train
    
    elif sampling_method == "smote":
        cat_indices = get_categorical_indices(X_train)
        smote = SMOTENC(categorical_features=cat_indices, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        return X_train, y_train
    
    elif sampling_method == "undersample":
        under = RandomUnderSampler(random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        return X_train, y_train
    
    elif sampling_method == "combined":
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        cat_indices = get_categorical_indices(X_train)
        smote = SMOTENC(categorical_features=cat_indices, sampling_strategy=1.0, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        return X_train, y_train
    
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")


# ======================================================
# 1. GLM tuning
# ======================================================

def tune_glm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    sampling_method: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tune logistic regression (GLM) hyperparameters using GridSearchCV.
    """
    X_train, y_train = apply_sampling(X_train, y_train, sampling_method)
    
    glm_pipe = make_glm_pipeline(
        schema=FINAL_SCHEMA,
        log_income=True,
        income_col="annual_income",
    )
    
    # Hyperparameter Grid
    param_grid = {
        "model__alpha": [1e-6, 1e-5, 1e-4, 1e-3],
        "model__l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
    }
        
    # Cross-validation
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )
    
    # Grid Search
    grid = GridSearchCV(
        estimator=glm_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ======================================================
# 2. LGBM tuning
# ======================================================

def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    sampling_method: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tune LightGBM hyperparameters using GridSearchCV.
    """
    X_train, y_train = apply_sampling(X_train, y_train, sampling_method)
    
    lgbm_pipe = make_lgbm_pipeline(
        schema=FINAL_SCHEMA,
        log_income=True,
        income_col="annual_income",
    )
    
    # Hyperparameter Grid
    param_grid = {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__n_estimators": [200, 400, 800],
        "model__num_leaves": [15, 31, 63],
        "model__min_child_weight": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
    }
    
    # Cross-validation
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )
    
    # Grid Search
    grid = GridSearchCV(
        estimator=lgbm_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ======================================================
# 3. Compare all sampling methods
# ======================================================

SAMPLING_METHODS = [None, "smote", "undersample", "combined"]

def run_tuning_all_sampling(
    model_type: str,
    X_train, y_train, X_val, y_val,
    cv_splits: int = 5,
):
    """
    Run tuning for baseline + 3 sampling methods.
    """
    if model_type not in ("glm", "lgbm"):
        raise ValueError("model_type must be 'glm' or 'lgbm'")

    tune_func = tune_glm if model_type == "glm" else tune_lgbm

    models = {}
    rows = []

    for method in SAMPLING_METHODS:
        method_name = "baseline" if method is None else method
        print("=" * 70)
        print(f"Tuning {model_type.upper()} | sampling={method_name}")
        print("=" * 70)

        best_model, best_params, best_cv_auc = tune_func(
            X_train, y_train,
            sampling_method=method,
            cv_splits=cv_splits
        )

        # validation AUC
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)

        models[method_name] = best_model
        rows.append({
            "model": model_type.upper(),
            "sampling": method_name,
            "cv_auc": best_cv_auc,
            "val_auc": val_auc,
            "best_params": str(best_params),
        })

        print(f"Best CV AUC: {best_cv_auc:.4f}")
        print(f"Validation AUC (tuned): {val_auc:.4f}")
        print(f"Best params: {best_params}\n")

    results_df = pd.DataFrame(rows).sort_values(["model", "val_auc"], ascending=[True, False])
    return models, results_df
