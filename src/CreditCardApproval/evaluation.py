from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal, Union, Mapping

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# ======================================================
# 1. Evaluation Factor Calculating
# ======================================================

@dataclass(frozen=True)
class EvalResult:
    """
    Container for evaluation results at a given threshold.
    """
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    auc: float
    ks: float
    gini: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
            "auc": self.auc,
            "ks": self.ks,
            "gini": self.gini,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def _predict_proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    """
    Get positive-class probabilities from a fitted model or pipeline.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must support predict_proba for AUC/KS/Gini evaluation.")
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("predict_proba output has unexpected shape.")
    return proba[:, 1]


def ks_statistic(y_true: pd.Series, y_score: np.ndarray) -> float:
    """
    Compute KS statistic: max(TPR - FPR) over ROC thresholds.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini_from_auc(auc: float) -> float:
    """
    Gini coefficient in credit scoring: Gini = 2*AUC - 1
    """
    return float(2.0 * auc - 1.0)


def evaluate_at_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
    zero_division: int = 0,
) -> EvalResult:
    """
    Evaluate metrics at a specific decision threshold.
    """
    y_pred = (y_score >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_score))
    ks = ks_statistic(y_true, y_score)
    gini = gini_from_auc(auc)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=zero_division))
    rec = float(recall_score(y_true, y_pred, zero_division=zero_division))
    f1 = float(f1_score(y_true, y_pred, zero_division=zero_division))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=zero_division))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return EvalResult(
        threshold=float(threshold),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        f2=f2,
        auc=auc,
        ks=ks,
        gini=gini,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


# ======================================================
# 2. Find Optimal Threshold
# ======================================================

def find_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    metric: Literal["f2", "f1", "recall", "precision", "accuracy", "youden", "profit"] = "f2",
    thresholds: Optional[np.ndarray] = None,
    beta: float = 2.0,
    
    # profit params (only used when metric="profit")
    tp_gain: float = 0.0,
    tn_gain: float = 0.0,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> Tuple[float, pd.DataFrame]:
    """
    Search the best decision threshold for an imbalanced binary classifier.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    rows = []

    # For Youden's J we can compute via roc_curve (more efficient & standard)
    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        # roc_curve returns thresholds in descending order incl. inf, we skip inf
        valid = np.isfinite(thr)
        fpr, tpr, thr = fpr[valid], tpr[valid], thr[valid]
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_threshold = float(thr[best_idx])

        # build a small table around all thresholds from roc_curve
        for _t, _fpr, _tpr, _j in zip(thr, fpr, tpr, j):
            y_pred = (y_proba >= _t).astype(int)
            rows.append(
                {
                    "threshold": float(_t),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
                    "youden": float(_j),
                    "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
                    "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
                    "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
                    "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
                }
            )

        df = pd.DataFrame(rows).sort_values("youden", ascending=False).reset_index(drop=True)
        return best_threshold, df

    # Generic grid search on thresholds
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)
        f2v = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

        profit = tp * tp_gain + tn * tn_gain - fp * fp_cost - fn * fn_cost

        row = {
            "threshold": float(t),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
            "f2": float(f2v),
            "profit": float(profit),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if metric == "f2":
        df = df.sort_values("f2", ascending=False)
    elif metric == "f1":
        df = df.sort_values("f1", ascending=False)
    elif metric == "recall":
        df = df.sort_values("recall", ascending=False)
    elif metric == "precision":
        df = df.sort_values("precision", ascending=False)
    elif metric == "accuracy":
        df = df.sort_values("accuracy", ascending=False)
    elif metric == "profit":
        df = df.sort_values("profit", ascending=False)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    df = df.reset_index(drop=True)
    best_threshold = float(df.loc[0, "threshold"])
    return best_threshold, df


# ======================================================
# 3. Compare Model
# ======================================================

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> EvalResult:
    """
    Evaluate a fitted model/pipeline on a dataset at a chosen threshold.
    """
    y_score = _predict_proba_positive(model, X)
    return evaluate_at_threshold(y_true=y, y_score=y_score, threshold=threshold)


def compare_models(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    threshold: Union[float, Mapping[str, float]] = 0.5,
) -> pd.DataFrame:
    """
    Evaluate multiple fitted models and return a summary table.
    """
    rows = []

    for name, model in models.items():
        t = threshold[name] if isinstance(threshold, dict) else threshold
        res = evaluate_model(model, X, y, threshold=t)
        d = res.to_dict()
        d["model"] = name
        rows.append(d)

    df = pd.DataFrame(rows).set_index("model")
    
    order = [
        "threshold",
        "auc",
        "gini",
        "ks",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    return df[order]


# ======================================================
# 4. Visualization
# ======================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Plot ROC Curve
def plot_roc_curves(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "ROC Curve Comparison",
):
    """
    Plot ROC curves for multiple fitted models.
    """
    plt.figure(figsize=(7, 5))

    for name, model in models.items():
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_score)
        auc = roc_auc_score(y, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    

# Plot Confusion Matrix
def plot_confusion_matrix(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    title: str | None = None,
):
    """
    Plot confusion matrix at a given threshold.
    """
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Good (0)", "Actual: Bad (1)"],
        columns=["Pred: Good (0)", "Pred: Bad (1)"],
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title if title else f"Confusion Matrix (threshold={threshold})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


# Plot Threshold Curves
def plot_threshold_curves(
    y_true,
    y_proba,
    beta: float = 2.0,
    thresholds: np.ndarray | None = None,
    title: str = "Threshold vs Metrics",
):
    """
    Plot metric curves as functions of decision threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    precision, recall, f1, f2 = [], [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        precision.append(precision_score(y_true, y_pred, zero_division=0))
        recall.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
        f2.append(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.plot(thresholds, f1, label="F1")
    plt.plot(thresholds, f2, label="F2", linewidth=2)

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot Predicted vs Actual
def _get_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return predicted probability of positive class (y==1).
    Works for sklearn Pipelines / estimators with predict_proba.
    """
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("predict_proba output shape is unexpected. Need proba for binary classification.")
    return proba[:, 1]


def plot_predicted_vs_actual(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
    title: str = "Predicted vs Actual (Calibration by Bins)",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Plot predicted vs actual default rate by probability bins (calibration-style plot).
    """
    y_true = pd.Series(y).astype(int).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfectly calibrated")

    rows = []

    for name, model in models.items():
        y_proba = _get_positive_proba(model, X)
        y_proba = pd.Series(y_proba, name="proba").reset_index(drop=True)

        df = pd.concat([y_true.rename("y"), y_proba], axis=1)

        if strategy == "quantile":
            df["bin"] = pd.qcut(df["proba"], q=n_bins, duplicates="drop")
        elif strategy == "uniform":
            df["bin"] = pd.cut(df["proba"], bins=n_bins)
        else:
            raise ValueError("strategy must be 'quantile' or 'uniform'.")

        g = (df.groupby("bin", observed=True).agg(
                n=("y", "size"),
                actual_rate=("y", "mean"),
                avg_pred=("proba", "mean"),
                min_pred=("proba", "min"),
                max_pred=("proba", "max"),
            )
            .reset_index()
        )

        g["model"] = name
        rows.append(g)

        ax.plot(g["avg_pred"], g["actual_rate"], marker="o", label=name)

    ax.set_xlabel("Average predicted probability")
    ax.set_ylabel("Actual positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return pd.concat(rows, ignore_index=True)


# ======================================================
# 5. Feature Importance
# ======================================================

def get_feature_names_from_pipeline(pipe) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    return list(pre.get_feature_names_out())


def get_glm_feature_importance(pipe, top_k: int = 20) -> pd.DataFrame:
    feature_names = get_feature_names_from_pipeline(pipe)
    model = pipe.named_steps["model"]
    coefs = model.coef_.ravel()

    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_k).reset_index(drop=True)
    return df

def get_lgbm_feature_importance(pipe, top_k: int = 20) -> pd.DataFrame:
    feature_names = get_feature_names_from_pipeline(pipe)
    model = pipe.named_steps["model"]
    importances = model.feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    return df
