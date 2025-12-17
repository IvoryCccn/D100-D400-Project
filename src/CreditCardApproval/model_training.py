import pandas as pd
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

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

def get_categorical_indices(X: pd.DataFrame) -> list:
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    return [X.columns.get_loc(col) for col in categorical_cols]


def train_glm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sampling_method: Optional[str] = None,
):
    """
    Train different logistic regression model and return AUC.
    """
    # choose different sampling method
    if sampling_method == "smote":
        cat_indices = get_categorical_indices(X_train)
        smote = SMOTENC(categorical_features=cat_indices, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After applying SMOTE, sample number: {len(y_train)}, positive proportion: {y_train.mean():.2%}")
    
    elif sampling_method == "undersample":
        under = RandomUnderSampler(random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        print(f"After applying Undersample, sample number: {len(y_train)}, positive proportion: {y_train.mean():.2%}")
    
    elif sampling_method == "combined":
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        cat_indices = get_categorical_indices(X_train)
        smote = SMOTENC(categorical_features=cat_indices, sampling_strategy=1.0, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After applying Combined, sample number: {len(y_train)}, positive proportion: {y_train.mean():.2%}")
    
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
    sampling_method: Optional[str] = None,
):
    """
    Train different LightGBM model and return AUC.
    """
    # choose different sampling method
    if sampling_method == "smote":
        cat_indices = get_categorical_indices(X_train)
        smote = SMOTENC(categorical_features=cat_indices, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After applying SMOTE, sample number: {len(y_train)}, positive proportion: {y_train.mean():.2%}")
    
    elif sampling_method == "undersample":
        under = RandomUnderSampler(random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        print(f"After applying Undersample, sample number: {len(y_train)}, positive proportion: {y_train.mean():.2%}")
    
    elif sampling_method == "combined":
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
        cat_indices = get_categorical_indices(X_train)
        smote = SMOTENC(categorical_features=cat_indices, sampling_strategy=1.0, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After applying Combined, sample number: {len(y_train)}, positive proportion: {y_train.mean():.2%}")
    
    lgbm_pipe = make_lgbm_pipeline(
        schema=FINAL_SCHEMA,
        log_income=True,
        income_col="annual_income",
    )

    lgbm_pipe.fit(X_train, y_train)
    val_proba = lgbm_pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    return lgbm_pipe, auc


# ======================================================
# 3. Aggregate all outcomes
# ======================================================

SAMPLING_METHODS = [None, "smote", "undersample", "combined"]

def run_training_all_sampling(
    model_type: str,
    X_train, y_train, X_val, y_val,
):
    """
    Run baseline + 3 sampling methods training.
    """
    if model_type not in ("glm", "lgbm"):
        raise ValueError("model_type must be 'glm' or 'lgbm'")
    
    train_func = train_glm if model_type == "glm" else train_lgbm

    models = {}
    rows = []

    for method in SAMPLING_METHODS:
        method_name = "baseline" if method is None else method
        print("=" * 70)
        print(f"Training {model_type.upper()} | sampling={method_name}")
        print("=" * 70)

        m, auc = train_func(
            X_train, y_train,
            X_val, y_val,
            sampling_method=method
        )

        models[method_name] = m
        rows.append({"model": model_type.upper(), "sampling": method_name, "val_auc": auc})
        print(f"Validation AUC = {auc:.4f}\n")

    auc_table = pd.DataFrame(rows).sort_values(["model", "val_auc"], ascending=[True, False])
    return models, auc_table