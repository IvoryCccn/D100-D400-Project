from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from CreditCardApproval.paths import get_data_dir


def get_data_paths(
    application_filename: str = "application_record.csv",
    credit_filename: str = "credit_record.csv",
) -> Tuple[Path, Path]:
    """
    Get full paths for application and credit datasets.

    Returns:
        (application_path, credit_path)
    """
    data_dir = get_data_dir()

    application_path = data_dir / application_filename
    credit_path = data_dir / credit_filename

    if not application_path.exists():
        raise FileNotFoundError(f"Application data not found: {application_path}")

    if not credit_path.exists():
        raise FileNotFoundError(f"Credit data not found: {credit_path}")

    return application_path, credit_path


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw application and credit datasets.

    Returns:
        application_df, credit_df
    """
    application_path, credit_path = get_data_paths()

    application_df = pd.read_csv(application_path)
    credit_df = pd.read_csv(credit_path)

    return application_df, credit_df
