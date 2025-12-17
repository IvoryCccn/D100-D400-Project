from __future__ import annotations

import pandas as pd
import pytest

from CreditCardApproval.data import get_data_paths, load_data


def test_get_data_paths_success(tmp_path, monkeypatch):
    # create dummy raw files
    (tmp_path / "application_record.csv").write_text("ID\n1\n")
    (tmp_path / "credit_record.csv").write_text("ID,MONTHS_BALANCE,STATUS\n1,0,C\n")

    # patch get_data_dir used inside CreditCardApproval.data
    monkeypatch.setattr("CreditCardApproval.data.get_data_dir", lambda: tmp_path)

    app_path, credit_path, processed_path = get_data_paths(require_processed=False)

    assert app_path.exists()
    assert credit_path.exists()
    assert app_path.name == "application_record.csv"
    assert credit_path.name == "credit_record.csv"
    # processed is allowed to not exist in this mode
    assert processed_path.name == "processed_df.parquet"


def test_get_data_paths_missing_application(tmp_path, monkeypatch):
    # only create credit file
    (tmp_path / "credit_record.csv").write_text("ID,MONTHS_BALANCE,STATUS\n1,0,C\n")
    monkeypatch.setattr("CreditCardApproval.data.get_data_dir", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        get_data_paths(require_processed=False)


def test_get_data_paths_missing_credit(tmp_path, monkeypatch):
    # only create application file
    (tmp_path / "application_record.csv").write_text("ID\n1\n")
    monkeypatch.setattr("CreditCardApproval.data.get_data_dir", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        get_data_paths(require_processed=False)


def test_get_data_paths_require_processed_raises(tmp_path, monkeypatch):
    # create only raw files
    (tmp_path / "application_record.csv").write_text("ID\n1\n")
    (tmp_path / "credit_record.csv").write_text("ID,MONTHS_BALANCE,STATUS\n1,0,C\n")

    monkeypatch.setattr("CreditCardApproval.data.get_data_dir", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        get_data_paths(require_processed=True)


def test_load_data_reads_raw_csvs(tmp_path, monkeypatch):
    app_df = pd.DataFrame({"ID": [1, 2], "CODE_GENDER": ["M", "F"]})
    credit_df = pd.DataFrame(
        {"ID": [1, 1, 2], "MONTHS_BALANCE": [0, -1, 0], "STATUS": ["C", "0", "X"]}
    )

    app_df.to_csv(tmp_path / "application_record.csv", index=False)
    credit_df.to_csv(tmp_path / "credit_record.csv", index=False)

    monkeypatch.setattr("CreditCardApproval.data.get_data_dir", lambda: tmp_path)

    app_loaded, credit_loaded = load_data(return_type="raw")

    assert app_loaded.shape == app_df.shape
    assert credit_loaded.shape == credit_df.shape
    assert list(app_loaded.columns) == list(app_df.columns)
    assert list(credit_loaded.columns) == list(credit_df.columns)


def test_load_data_reads_processed_parquet(tmp_path, monkeypatch):
    # create raw files so get_data_paths always passes raw checks
    (tmp_path / "application_record.csv").write_text("ID\n1\n")
    (tmp_path / "credit_record.csv").write_text("ID,MONTHS_BALANCE,STATUS\n1,0,C\n")

    # create a minimal parquet file
    processed = pd.DataFrame({"ID": [1], "target": [0]})
    processed.to_parquet(tmp_path / "processed_df.parquet", index=False)

    monkeypatch.setattr("CreditCardApproval.data.get_data_dir", lambda: tmp_path)

    processed_loaded = load_data(return_type="processed")

    assert isinstance(processed_loaded, pd.DataFrame)
    assert processed_loaded.shape[0] == 1
    assert "ID" in processed_loaded.columns
    assert "target" in processed_loaded.columns