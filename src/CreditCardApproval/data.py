from __future__ import annotations

from pathlib import Path
from typing import Tuple
from typing import Literal, Union

import pandas as pd

from CreditCardApproval.paths import get_data_dir


def get_data_paths(
    application_filename: str = "application_record.csv",
    credit_filename: str = "credit_record.csv",
    processed_filename: str = "processed_df.parquet",
    require_processed: bool = False,
) -> Tuple[Path, Path, Path]:
    """
    Get full paths for raw and processed datasets.
    
    If require_processed=False (default), processed_path may not exist yet.
    """
    data_dir = get_data_dir()

    # raw data paths
    application_path = data_dir / application_filename
    credit_path = data_dir / credit_filename
    
    # processed data path
    processed_path = data_dir / processed_filename
    
    # ensure raw data exists
    if not application_path.exists():
        raise FileNotFoundError(f"Application data not found: {application_path}")

    if not credit_path.exists():
        raise FileNotFoundError(f"Credit data not found: {credit_path}")
    
    # optional processed data check
    if require_processed and (not processed_path.exists()):
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    return application_path, credit_path, processed_path


def load_data(
    return_type: Literal["raw", "processed", "all"] = "raw",
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame],
    pd.DataFrame,
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    """
    Load raw and/or processed datasets.

    return_type:
        - "raw": return (application_df, credit_df)
        - "processed": return processed_df
        - "all": return (application_df, credit_df, processed_df)
    """
    application_path, credit_path, processed_path = get_data_paths()

    application_df = credit_df = processed_df = None

    # load raw data
    if return_type in ("raw", "all"):
        application_df = pd.read_csv(application_path)
        credit_df = pd.read_csv(credit_path)

    # load processed data
    if return_type in ("processed", "all"):
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_path}")

        processed_df = pd.read_parquet(processed_path)

    # return
    if return_type == "raw":
        return application_df, credit_df

    if return_type == "processed":
        return processed_df

    return application_df, credit_df, processed_df
