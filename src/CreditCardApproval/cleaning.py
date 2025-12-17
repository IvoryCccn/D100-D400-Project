import pandas as pd


APPLICATION_RENAME_MAP = {
    "AMT_INCOME_TOTAL": "annual_income",
    "CNT_CHILDREN": "children_number",
    "CNT_FAM_MEMBERS": "family_size",

    "CODE_GENDER": "gender",
    "FLAG_OWN_CAR": "own_car",
    "FLAG_OWN_REALTY": "own_realty",
    "FLAG_WORK_PHONE": "own_work_phone",
    "FLAG_PHONE": "own_phone",
    "FLAG_EMAIL": "own_email",

    "NAME_INCOME_TYPE": "income_type",
    "NAME_EDUCATION_TYPE": "education_type",
    "NAME_FAMILY_STATUS": "family_status",
    "NAME_HOUSING_TYPE": "housing_type",
}


def rename_application_columns(
    df: pd.DataFrame,
    rename_map: dict = APPLICATION_RENAME_MAP,
) -> pd.DataFrame:
    """
    Rename selected application features to snake_case for modelling.
    """
    df = df.copy()
    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}

    return df.rename(columns=existing_map)


def handle_application_missing(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing values in application data.

    Handling strategy:
        - employed_years: set to 0 for unemployed applicants (is_unemployed == 1) while keep original value for employed applicants
    """
    df = df.copy()

    if "employed_years" in df.columns and "is_unemployed" in df.columns:
        df.loc[df["is_unemployed"] == 1, "employed_years"] = 0

    return df


def clip_outliers_application(
    df: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Clip outliers of numerical variables in application data
    based on EDA findings.

    Clipping strategy:
        - annual_income: clip at 1% and 99%
        - employed_years: clip at 99%
        - children_number, family_size: clip at 99%
    """
    df = df.copy()

    # 1) Annual income: strong right skew (two-sided)
    if "annual_income" in df.columns:
        lo, hi = df["annual_income"].quantile([lower_quantile, upper_quantile])
        df["annual_income"] = df["annual_income"].clip(lo, hi)

    # 2) Employed years: long right tail (upper only)
    if "employed_years" in df.columns:
        hi = df["employed_years"].quantile(upper_quantile)
        df["employed_years"] = df["employed_years"].clip(upper=hi)

    # 3) Children number: upper only
    if "children_number" in df.columns:
        hi = df["children_number"].quantile(upper_quantile)
        df["children_number"] = df["children_number"].clip(upper=hi)

    # 4) Family size: upper only
    if "family_size" in df.columns:
        hi = df["family_size"].quantile(upper_quantile)
        df["family_size"] = df["family_size"].clip(upper=hi)

    return df


def recode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode categorical variables based on EDA findings to reduce sparsity and improve model stability.
    """
    df = df.copy()

    # Income type
    income_mapping = {
        "Working": "Employed",
        "Commercial associate": "Employed",
        "State servant": "Employed",
        "Businessman": "Self-Employed",
        "Pensioner": "Retired",
        "Student": "Other",
        "Unemployed": "Other",
        "Maternity leave": "Other",
    }
    df["income_type_recoded"] = df["income_type"].map(income_mapping)

    # Education level
    education_mapping = {
        "Higher education": "Higher",
        "Academic degree": "Higher",
        "Incomplete higher": "Secondary",
        "Secondary / secondary special": "Secondary",
        "Lower secondary": "Lower",
    }
    df["education_level"] = df["education_type"].map(education_mapping)

    # Family status
    family_mapping = {
        "Married": "Married",
        "Civil marriage": "Married",
        "Single / not married": "Single",
        "Separated": "Previously_Married",
        "Widow": "Previously_Married",
    }
    df["family_status_recoded"] = df["family_status"].map(family_mapping)

    # Housing type
    housing_mapping = {
        "House / apartment": "Own_Home",
        "Co-op apartment": "Own_Home",
        "Municipal apartment": "Rental",
        "Rented apartment": "Rental",
        "Office apartment": "Rental",
        "With parents": "With_Parents",
    }
    df["housing_status"] = df["housing_type"].map(housing_mapping)

    return df
