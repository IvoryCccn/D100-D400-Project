from pathlib import Path

from CreditCardApproval.paths import (
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    get_figures_dir,
    get_models_dir,
)


def test_project_root_exists():
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert root.is_dir()


def test_data_dir():
    data_dir = get_data_dir()
    assert data_dir.name == "data"
    assert data_dir.exists()
    assert data_dir.is_dir()


def test_outputs_dir_created():
    outputs_dir = get_outputs_dir()
    assert outputs_dir.exists()
    assert outputs_dir.is_dir()
    assert outputs_dir.name == "outputs"


def test_figures_dir_created():
    figures_dir = get_figures_dir()
    assert figures_dir.exists()
    assert figures_dir.is_dir()
    assert figures_dir.name == "figures"


def test_models_dir_created():
    models_dir = get_models_dir()
    assert models_dir.exists()
    assert models_dir.is_dir()
    assert models_dir.name == "models"
