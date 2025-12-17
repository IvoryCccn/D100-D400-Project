from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Return the project root directory.
    Assumes this file lives in: <project_root>/src/CreditCardApproval/paths.py
    """
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the data directory under project root."""
    return get_project_root() / "data"


def get_outputs_dir() -> Path:
    """Return outputs directory under project root."""
    out = get_project_root() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_figures_dir() -> Path:
    """Return outputs/figures directory under project root."""
    fig_dir = get_outputs_dir() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def get_models_dir() -> Path:
    """Return outputs/models directory under project root."""
    model_dir = get_outputs_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
