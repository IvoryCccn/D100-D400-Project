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


def clip_outliers_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outliers of numerical variables in application data
    based on EDA findings.

    Clipping strategy:
        - annual_income: clip at 1% and 99%
        - employed_years: clip at 99%
        - children_number, family_size: clip at 99%
    """
    df = df.copy()

    # 1. Annual income: strong right skew
    if "annual_income" in df.columns:
        lower, upper = df["annual_income"].quantile([0.01, 0.99])
        df["annual_income"] = df["annual_income"].clip(lower, upper)

    # 2. Employed years: long right tail
    if "employed_years" in df.columns:
        upper = df["employed_years"].quantile(0.99)
        df["employed_years"] = df["employed_years"].clip(upper=upper)

    # 3. Children number
    if "children_number" in df.columns:
        upper = df["children_number"].quantile(0.99)
        df["children_number"] = df["children_number"].clip(upper=upper)

    # 4. Family size
    if "family_size" in df.columns:
        upper = df["family_size"].quantile(0.99)
        df["family_size"] = df["family_size"].clip(upper=upper)

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
        "Businessman": "Self_Employed",
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


def merge_application_with_target(
    application_df: pd.DataFrame,
    target_df: pd.DataFrame,
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Merge cleaned application data with credit target.
    Only applicants with credit history are retained.
    """
    merged = application_df.merge(
        target_df[[id_col, "target"]],
        on=id_col,
        how="inner",
    )

    return merged


def select_final_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select final variables set for modeling.
    Drops original high-cardinality categorical columns.
    """
    df = df.copy()
    
    drop_cols = ["income_type", "education_type", "family_status", "housing_type"]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    df = df.rename(columns={"family_status_recoded": "family_status", "income_type_recoded": "income_type"})

    return df
