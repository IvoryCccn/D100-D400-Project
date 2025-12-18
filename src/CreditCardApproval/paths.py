from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Locate project root from current file path.
    """
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return data directory """
    return get_project_root() / "data"


def get_outputs_dir() -> Path:
    """Create and return outputs directory."""
    out = get_project_root() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_figures_dir() -> Path:
    """Create and return figures directory."""
    fig_dir = get_outputs_dir() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def get_models_dir() -> Path:
    """Create and return models directory."""
    model_dir = get_outputs_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
