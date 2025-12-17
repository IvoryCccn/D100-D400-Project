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

from CreditCardApproval.paths import get_models_dir


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

    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        valid = np.isfinite(thr)
        fpr, tpr, thr = fpr[valid], tpr[valid], thr[valid]
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_threshold = float(thr[best_idx])

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
# 3. Visualization
# ======================================================

import matplotlib.pyplot as plt
import seaborn as sns
import math

# Plot ROC Curve
def plot_roc_curves_grid(
    models: dict,
    X,
    y,
    title: str = "ROC Curves (Validation)",
    save_name: str | None = None,
    ncols: int = 4,
):
    names = list(models.keys())
    n = len(names)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(names):
        ax = axes[i]
        model = models[name]
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_score)
        auc = roc_auc_score(y, y_score)

        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

        ax.set_title(f"{name}\nAUC={auc:.3f}", fontsize=10)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_name:
        save_path = get_models_dir() / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Plot Confusion Matrix
def plot_confusion_matrices_grid(
    models: dict,
    X,
    y,
    thresholds: dict,
    title: str = "Confusion Matrices @ Optimal Threshold (Validation)",
    save_name: str | None = None,
    ncols: int = 4,
):
    names = list(models.keys())
    n = len(names)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(names):
        ax = axes[i]
        model = models[name]
        t = float(thresholds[name])

        y_score = model.predict_proba(X)[:, 1]
        y_pred = (y_score >= t).astype(int)

        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Good (0)", "Actual: Bad (1)"],
            columns=["Pred: Good (0)", "Pred: Bad (1)"],
        )

        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{name}\nthr={t:.2f}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_name:
        save_path = get_models_dir() / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Plot Threshold Curves
def plot_threshold_curves_grid(
    models: dict,
    X,
    y,
    optimal_thresholds: dict | None = None,
    beta: float = 2.0,
    thresholds: np.ndarray | None = None,
    title: str = "Threshold Curves (Validation)",
    save_name: str | None = None,
    ncols: int = 4,
):
    names = list(models.keys())
    n = len(names)
    nrows = math.ceil(n / ncols)

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(names):
        ax = axes[i]
        model = models[name]
        y_proba = model.predict_proba(X)[:, 1]

        precision, recall, f1, f2 = [], [], [], []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            precision.append(precision_score(y, y_pred, zero_division=0))
            recall.append(recall_score(y, y_pred, zero_division=0))
            f1.append(f1_score(y, y_pred, zero_division=0))
            f2.append(fbeta_score(y, y_pred, beta=beta, zero_division=0))

        ax.plot(thresholds, precision, label="Precision", linewidth=1)
        ax.plot(thresholds, recall, label="Recall", linewidth=1)
        ax.plot(thresholds, f1, label="F1", linewidth=1)
        ax.plot(thresholds, f2, label=f"F{int(beta)}", linewidth=2)

        # notify optimal threshold
        if optimal_thresholds is not None and name in optimal_thresholds:
            t_star = float(optimal_thresholds[name])
            ax.axvline(t_star, linestyle="--", linewidth=1)
            ax.set_title(f"{name}\nthr={t_star:.2f}", fontsize=10)
        else:
            ax.set_title(name, fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_name:
        save_path = get_models_dir() / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# ======================================================
# 4. Compare Model
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


def evaluate_models_table(
    models: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: Union[float, Mapping[str, float]] = 0.5,
    sort_by: str | None = "auc",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Evaluate multiple fitted models and return a summary table.
    """
    df = compare_models(models=models, X=X, y=y, threshold=thresholds).reset_index()
    df = df.rename(columns={"model": "model_name"})

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    return df


# Plot Predicted vs Actual
def _get_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("predict_proba output shape is unexpected for binary classification.")
    return proba[:, 1]

def plot_predicted_vs_actual(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
    title: str = "Predicted vs Actual (Calibration by Bins)",
    save_name: str | None = None,
) -> pd.DataFrame:
    """
    Calibration-style plot: average predicted proba vs actual positive rate across bins.
    Returns a long-form table with bin stats for all models.
    """
    y_true = pd.Series(y).astype(int).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfectly calibrated")

    rows = []

    for name, model in models.items():
        y_proba = pd.Series(_get_positive_proba(model, X), name="proba").reset_index(drop=True)
        df = pd.concat([y_true.rename("y"), y_proba], axis=1)

        if strategy == "quantile":
            df["bin"] = pd.qcut(df["proba"], q=n_bins, duplicates="drop")
        elif strategy == "uniform":
            df["bin"] = pd.cut(df["proba"], bins=n_bins)
        else:
            raise ValueError("strategy must be 'quantile' or 'uniform'.")

        g = (
            df.groupby("bin", observed=True)
              .agg(
                  n=("y", "size"),
                  actual_rate=("y", "mean"),
                  avg_pred=("proba", "mean"),
                  min_pred=("proba", "min"),
                  max_pred=("proba", "max"),
              )
              .reset_index()
        )
        g["model"] = name
        g["n_bins_used"] = g.shape[0]
        rows.append(g)

        ax.plot(g["avg_pred"], g["actual_rate"], marker="o", label=f"{name}")

    ax.set_xlabel("Average predicted probability")
    ax.set_ylabel("Actual positive rate")
    ax.set_title(title + f" | bins={n_bins}, strategy={strategy}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_name:
        save_path = get_models_dir() / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

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
