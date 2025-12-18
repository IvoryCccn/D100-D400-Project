# D100-D400-Project
D100 and D400 Final Project (BGN 6142P)

# ======================================================
# Pre Setup
# ======================================================

Before running notebook, please ensure virtual environment is created from the terminal

    - cd <project_path>
    - conda create -n credit-card-approval
    - conda activate credit-card-approval
    - pip install -e .


# ======================================================
# Raw data
# ======================================================

data can be downloaded from https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data


# ======================================================
# Tool: function_usage_chech.py
# ======================================================

`function_usage_check.py` is a utility script used to analyze function usage across the project. It helps identify functions that are defined but never called, as well as potential missing or redundant logic in the codebase, by checking how functions are used in both `src` and `notebook` directories.

The script is placed in the parent (project root) directory and can be executed directly from the terminal. To run the check and view the results, simply use:

    - py function_usage_check.py

Since `.ipynb` files cannot be analyzed directly, the notebook folder contains Python (`.py`) versions of the corresponding notebooks. These converted files are used to ensure that function calls inside notebooks are also included in the usage check.


# ======================================================
# pytest
# ======================================================

To run a pytest for the project, please run the following in the terminal:

    - pytest


# ======================================================
# Project Mindmap
# ======================================================

D100-D400-Project
│  .gitignore
│  .pre-commit-config.yaml
│  environment.yml
│  function_usage_check.py
│  pyproject.toml
│  README.md
│
├─data
│      application_record.csv
│      credit_record.csv
│      processed_df.parquet
│
├─notebooks
│  │  eda_cleaning.ipynb
│  │  eda_cleaning.py
│  │  run_pipeline.ipynb
│  └─ run_pipeline.py
│
├─outputs
│  ├─figures
│  │      application_missing_matrix.png
│  │      boxplots_numerical_variables (Cleaned).png
│  │      boxplots_numerical_variables.png
│  │      credit_missing_matrix.png
│  │      credit_severity_counts.png
│  │      credit_severity_ratio.png
│  │      credit_status_counts.png
│  │      credit_status_ratio.png
│  │      distribution_binary_bar_charts.png
│  │      distribution_binary_pie_charts.png
│  │      distribution_birth_and_employed.png
│  │      distribution_creditrecords_ID.png
│  │      distribution_creditrecords_month.png
│  │      distribution_histograms.png
│  │      distribution_multiclass_bar_charts.png
│  │      heatmap_numercial_variables.png
│  │
│  └─models
│          16_Confusion_Matrices.png
│          16_ROC_Curves.png
│          16_Threshold_Curves.png
│          2_predicted_actual_calibration.png
│
├─src
│  └─CreditCardApproval
│     │  cleaning.py
│     │  data.py
│     │  eda.py
│     │  evaluation.py
│     │  modeling.py
│     │  model_training.py
│     │  paths.py
│     │  tuning.py
│     │  __init__.py
│     │
│     └─feature_engineering
│        │  my_transformer.py
│        └─ __init__.py
│ 
└─tests
   │  test_cleaning.py
   │  test_data.py
   │  test_evaluation.py
   │  test_modeling.py
   │  test_model_training.py
   │  test_my_transformer.py
   └─ test_paths.py
