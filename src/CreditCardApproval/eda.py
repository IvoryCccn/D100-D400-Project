from __future__ import annotations

from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

from CreditCardApproval.paths import get_figures_dir


# ======================================================
# 1. Basic summaries
# ======================================================

def basic_overview(df: pd.DataFrame, n_head: int = 5) -> None:
    """
    Print basic information for quick EDA.
    """
    print("\nNumber of datapoints for application records: {}".format(len(df)))
    print("\nNumber of unique clients in dataset: {}".format(len(df['ID'].unique())))
    print("\nShape:", df.shape)
    print("\nInformation:", df.info())
    return None


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150):
    """
    Save matplotlib figure to outputs/figures directory.
    """
    path = get_figures_dir() / filename
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ======================================================
# 2. Missing value
# ======================================================

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table of missing counts and missing rates by column.
    """
    miss_count = df.isna().sum()
    miss_rate = miss_count / len(df)
    out = (
        pd.DataFrame(
            {"missing_count": miss_count, "missing_rate": miss_rate}
        )
        .sort_values("missing_rate", ascending=False)
    )
    return out


def plot_missing_matrix(
    df: pd.DataFrame,
    title: str = "Missing data by column",
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot missing value bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    msno.matrix(df, ax=ax, fontsize=12, sparkline=False)
    
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')

    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.close()
    return fig


# ======================================================
# 3. Data type analysis
# ======================================================

def unique_value_summary(df: pd.DataFrame, ascending: bool = True, top_n: int = None) -> pd.DataFrame:
    """
    Calculate the number of unique values for each column.
    
    Returns DataFrame with columns ['Column_Name', 'Num_Unique'] sorted by unique count.
    """
    unique_counts = pd.DataFrame.from_records(
        [(col, df[col].nunique()) for col in df.columns],
        columns=['Column_Name', 'Num_Unique']
    ).sort_values(by='Num_Unique', ascending=ascending)
    
    if top_n is not None:
        unique_counts = unique_counts.head(top_n)
    
    return unique_counts


def split_num_cat_features(df: pd.DataFrame) -> tuple:
    """
    Simple split based on data types only.
    """
    numerical_features = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    numerical_df = df[numerical_features]
    categorical_df = df[['ID'] + categorical_features]

    return numerical_df, categorical_df


def describe_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive statistics for numerical features.
    """
    features = [col for col in df.columns if col != 'ID']

    stats = df[features].describe().T
    stats['skewness'] = df[features].skew()
    stats['kurtosis'] = df[features].kurtosis()
    
    return stats


def describe_categorical_features(df: pd.DataFrame) -> None:
    """
    Display value counts for categorical features.
    """
    features = [col for col in df.columns if col != 'ID']

    for col in features:
        print(f"\n{'='*60}")
        print(f"Column: {col}")
        print(f"{'='*60}")
        print(f"Number of unique values: {df[col].nunique()}")
        print(f"\nValue counts:")
        print(df[col].value_counts())

    return None


# ======================================================
# 4.1. Distribution analysis
# ======================================================

def plot_histograms(
    df: pd.DataFrame,
    columns: List[str],
    bins: int = 30,
    figsize: tuple = (12,8),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot histograms for multiple numerical variables in a 2x3 grid.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        data = df[col].dropna()

        ax.hist(data, bins=bins)
        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.set_xlabel(col)

    # Remove unused subplots
    if len(columns) < len(axes):
        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name)

    plt.show()


def plot_pie_charts(
    df: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (12, 4),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot pie charts for binary categorical variables in a single row.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=figsize)

    for ax, col in zip(axes, columns):
        counts = df[col].value_counts(dropna=False)
        ax.pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(col)

    if title:
        fig.suptitle(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name)

    plt.show()


def plot_binary_bar_charts(
    df: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (12, 4),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot bar charts for binary variables in a single row.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=figsize)

    for ax, col in zip(axes, columns):
        counts = df[col].value_counts(dropna=False).sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    if title:
        fig.suptitle(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name)

    plt.show()


def plot_multiclass_bar_charts(
    df: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (12, 8),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot bar charts for multi-class categorical variables in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        counts = df[col].value_counts(dropna=False)
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(col)
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    if title:
        fig.suptitle(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name)

    plt.show()


# ======================================================
# 4.2. Solving DAYS_BIRTH and DAYS_EMPLOYED
# ======================================================

def transform_birth_and_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform DAYS_BIRTH and DAYS_EMPLOYED into interpretable features.

    Outputs:
    - age_years: positive age in years
    - is_unemployed: 1 if DAYS_EMPLOYED is positive (special coding), else 0
    - employed_years: years employed for employed people; NaN for unemployed
    """
    out = df[["ID", "DAYS_BIRTH", "DAYS_EMPLOYED"]].copy()

    # handle DAYS_BIRTH
    out["age_years"] = (-out["DAYS_BIRTH"] / 365.25).astype(float)

    # handle DAYS_EMPLOYED
    out["is_unemployed"] = (out["DAYS_EMPLOYED"] > 0).astype(int)

    # For employed people (DAYS_EMPLOYED <= 0), convert to positive years
    out["employed_years"] = np.where(
        out["DAYS_EMPLOYED"] <= 0,
        (-out["DAYS_EMPLOYED"] / 365.25),
        np.nan,
    ).astype(float)

    return out


def plot_birth_and_employed(
    df: pd.DataFrame,
    age_col: str = "age_years",
    unemployed_col: str = "is_unemployed",
    employed_years_col: str = "employed_years",
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 4),
    title: str | None = None,
    save_name: str | None = None,
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
):
    """
    Plot 3 EDA charts in a single row (1x3):
    1) Age distribution histogram
    2) Employment status bar chart (0=employed, 1=unemployed)
    3) Employment duration histogram for employed people (optionally clipped for readability)

    Assumes the input df already contains:
    - age_col: positive age in years
    - unemployed_col: binary indicator (1 means unemployed)
    - employed_years_col: years employed (NaN for unemployed)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) Age histogram
    age_data = df[age_col].dropna()
    axes[0].hist(age_data, bins=bins)
    axes[0].set_title(age_col)
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")

    # 2) Unemployment status bar
    status_counts = df[unemployed_col].value_counts(dropna=False).sort_index()
    axes[1].bar(status_counts.index.astype(str), status_counts.values)
    axes[1].set_title(unemployed_col)
    axes[1].set_xlabel("Value (0=Employed, 1=Unemployed)")
    axes[1].set_ylabel("Count")

    # 3) Employment duration histogram (nested helper)
    def _plot_employed_years_hist(ax: plt.Axes) -> None:
        emp = df.loc[df[unemployed_col] == 0, employed_years_col].dropna()

        if emp.empty:
            ax.text(
                0.5, 0.5,
                "No employed records to plot.",
                ha="center", va="center"
            )
            ax.set_title(employed_years_col)
            ax.set_xlabel("Years employed")
            ax.set_ylabel("Count")
            return

        q_low, q_high = clip_quantiles
        if q_low is not None and q_high is not None:
            low = emp.quantile(q_low)
            high = emp.quantile(q_high)
            emp = emp.clip(lower=low, upper=high)
            subtitle = f"(clipped {int(q_low*100)}%-{int(q_high*100)}%)"
        else:
            subtitle = ""

        ax.hist(emp, bins=bins)
        ax.set_title(f"{employed_years_col} {subtitle}".strip())
        ax.set_xlabel("Years employed")
        ax.set_ylabel("Count")

    _plot_employed_years_hist(axes[2])

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    plt.show()


# ======================================================
# 5. Outliers analysis
# ======================================================

def plot_numeric_boxplots(
    df: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (12, 8),
    title: str | None = None,
    showfliers: bool = True,
    save_name: str | None = None,
):
    """
    Plot boxplots for selected numerical variables in a 2x3 grid.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        df[[col]].dropna().boxplot(
            ax=ax,
            showfliers=showfliers,
        )
        ax.set_title(col)
        ax.set_ylabel(col)

    # Hide unused subplots
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()


# ======================================================
# 6. Correlation analysis
# ======================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (8, 8),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot a Pearson correlation heatmap for selected numerical variables.
    """
    corr_matrix = df[columns].corr(method="pearson")

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if title:
        ax.set_title(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()


# ======================================================
# 7. Panal data analysis
# ======================================================

def plot_credit_records_per_id(
    df: pd.DataFrame,
    id_col: str = "ID",
    bins: int = 50,
    figsize: tuple = (6, 4),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot the distribution of number of credit record per ID.
    """
    counts = df.groupby(id_col).size()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(counts, bins=bins)
    ax.set_xlabel("Number of monthly credit records")
    ax.set_ylabel("Number of applicants")

    if title:
        ax.set_title(title)

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()

    return counts


def plot_months_balance_distribution(
    df: pd.DataFrame,
    month_col: str = "MONTHS_BALANCE",
    bins: int = 50,
    figsize: tuple = (6, 4),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot distribution of MONTHS_BALANCE.
    """
    months_ago = -df[month_col]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(months_ago, bins=bins)
    ax.set_xlabel("Months before current")
    ax.set_ylabel("Number of records")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Distribution of Credit Records Over Time")

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()


# ======================================================
# 8. Credit 'STATUS' analysis
# ======================================================

def plot_status_distribution(
    df: pd.DataFrame,
    status_col: str = "STATUS",
    figsize: tuple = (8, 4),
    title: str | None = None,
    normalize: bool = False,
    save_name: str | None = None,
):
    """
    Plot distribution of credit STATUS values.
    If parameters 'normalize' equal to True, plot proportions instead of counts.
    """
    counts = df[status_col].value_counts(dropna=False, normalize=normalize)
    counts = counts.sort_index()  # keep X/C/0..5 in stable order if possible

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title if title else "STATUS Distribution")
    ax.set_xlabel("STATUS")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.tick_params(axis="x", rotation=0)

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()

    return counts


# Higher value = worse credit behavior, according to https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data
STATUS_SEVERITY = {
    "X": 0,   # no loan for the month
    "C": 0,   # paid off that month
    "0": 1,   # 1-29 days past due
    "1": 2,   # 30-59 days overdue
    "2": 3,   # 60-89 days overdue
    "3": 4,   # 90-119 days overdue
    "4": 5,   # 120-149 days overdue
    "5": 6,   # overdue or bad debts, write-offs for more than 150 days
}


def add_status_severity(
    df: pd.DataFrame,
    status_severity: Dict[str, int],
    status_col: str = "STATUS",
    new_col: str = "severity",
) -> pd.DataFrame:
    """
    Add a severity column mapped from STATUS.
    """
    out = df.copy()
    out[new_col] = out[status_col].astype(str).map(status_severity)
    return out


def plot_severity_distribution(
    df: pd.DataFrame,
    severity_col: str = "severity",
    figsize: tuple = (8, 4),
    title: str | None = None,
    normalize: bool = False,
    save_name: str | None = None,
):
    """
    Plot distribution of mapped severity values.
    If parameters 'normalize' equal to True, plot proportions instead of counts.
    """
    counts = df[severity_col].value_counts(dropna=False, normalize=normalize).sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title if title else "Severity Distribution")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.tick_params(axis="x", rotation=0)

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()

    return counts


# ======================================================
# 9. ID-level aggregation
# ======================================================

def aggregate_credit_by_id(
    df: pd.DataFrame,
    id_col: str = "ID",
    severity_col: str = "severity",
) -> pd.DataFrame:
    """
    Aggregate monthly credit records to customer (ID) level.

    Returns:
        pd.DataFrame with ID-level aggregated credit features:
            - n_months: number of observed months
            - max_severity: worst observed severity
            - mean_severity: average severity
            - severe_count: number of months with severity >= 3
    """
    agg_df = (
        df.groupby(id_col)[severity_col]
        .agg(
            n_months="count",
            max_severity="max",
            mean_severity="mean",
            severe_count=lambda x: (x >= 3).sum(),
        )
        .reset_index()
    )

    return agg_df


def plot_credit_history_length(
    agg_df: pd.DataFrame,
    length_col: str = "n_months",
    bins: int = 50,
    figsize: tuple = (6, 4),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot distribution of credit history length per applicant.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(agg_df[length_col], bins=bins)
    ax.set_xlabel("Number of months observed")
    ax.set_ylabel("Number of applicants")
    ax.set_title(title if title else "Credit History Length per Applicant")

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()


def plot_max_severity_distribution(
    agg_df: pd.DataFrame,
    severity_col: str = "max_severity",
    figsize: tuple = (6, 4),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot distribution of maximum severity per applicant.
    """
    counts = agg_df[severity_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Maximum severity observed")
    ax.set_ylabel("Number of applicants")
    ax.set_title(title if title else "Maximum Credit Severity per Applicant")

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()

    return counts


def plot_ever_severe_pie(
    agg_df: pd.DataFrame,
    severe_col: str = "severe_count",
    figsize: tuple = (6, 4),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot whether applicants have ever experienced severe delinquency (severity >= 3).
    """
    labels = ["Never severe delinquency", "At least one severe delinquency"]
    values = [
        (agg_df[severe_col] == 0).sum(),
        (agg_df[severe_col] > 0).sum(),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title if title else "Ever Experienced Severe (>=3) Delinquency")

    if save_name:
        save_figure(fig, save_name)

    plt.tight_layout()
    plt.show()


# ======================================================
# 10. Target definition
# ======================================================

def build_credit_target(
    credit_df: pd.DataFrame,
    status_severity: dict,
    id_col: str = "ID",
    status_col: str = "STATUS",
    bad_severity_threshold: int = 3,
) -> pd.DataFrame:
    """
    Construct customer-level credit target from monthly credit records.

    Returns:
        pd.DataFrame with columns:
            - ID
            - target (1 = bad, 0 = good)
            - max_severity
            - total_months_per_ID
    """
    df = credit_df.copy()
    df["severity"] = df[status_col].astype(str).map(status_severity)

    agg = (
        df.groupby(id_col)["severity"]
        .agg(
            max_severity="max",
            total_months_per_ID="count",
        )
        .reset_index()
    )

    agg["target"] = (agg["max_severity"] >= bad_severity_threshold).astype(int)

    return agg
