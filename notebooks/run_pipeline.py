#!/usr/bin/env python
# coding: utf-8

# Note: this notebook aims to run the pipeline of modeling, model_training, model_tuning and model_evaluation, and make conclusion in the end.

# ### 1. Data Loading

# In[1]:


import pandas as pd
from CreditCardApproval.data import load_data

processed_df = load_data(return_type="processed")


# In[2]:


# brief look
print("processed_df shape:", processed_df.shape)
processed_df.head()


# In[3]:


print("Is ID unique:", processed_df["ID"].is_unique)
processed_df.info()


# ***

# ### 2. Modeling

# ##### ① Specify X and y

# In[4]:


from CreditCardApproval.modeling import make_X_y, summarize_target

X, y = make_X_y(processed_df)
summarize_target(y)


# Review:
# 
# The target variable is highly imbalanced, which is typical in credit risk modeling. Instead of resampling the data, class imbalance is handled at the modeling stage by using class-weighted loss functions and appropriate evaluation metrics. Logistic regression applies balanced class weights, while LightGBM incorporates scale_pos_weight computed from the training data. Model performance is evaluated using AUC to ensure robustness under imbalance.

# ##### ② Configuration and feature schema

# In[5]:


from CreditCardApproval.modeling import FINAL_SCHEMA, validate_schema

validate_schema(processed_df, FINAL_SCHEMA)


# ##### ③ Build up model pipeline

# In[6]:


from CreditCardApproval.modeling import make_glm_pipeline, make_lgbm_pipeline

glm_pipe = make_glm_pipeline(
    schema=FINAL_SCHEMA,
    log_income=True,
    income_col="annual_income",
)
glm_pipe


# In[7]:


lgbm_pipe = make_lgbm_pipeline(
    schema=FINAL_SCHEMA,
    log_income=True,
    income_col="annual_income",
)
lgbm_pipe


# ##### ④ Preview pipeline dimension

# In[8]:


from CreditCardApproval.modeling import preview_pipeline_dimensions

glm_preview = preview_pipeline_dimensions(glm_pipe, X)
print("GLM model pipeline preview:")
glm_preview


# In[9]:


lgbm_preview = preview_pipeline_dimensions(lgbm_pipe, X)
print("LGBM model pipeline preview:")
lgbm_preview


# Review:
# 
# The preview confirms that both the GLM and LGBM pipelines produce the same feature dimensionality after preprocessing. This indicates that feature engineering steps, including categorical encoding and numerical transformations, are applied consistently across models, ensuring a fair comparison in subsequent training and evaluation.

# ***

# ### 3. Model Training

# The aim of this part is to train model based on specified parameter, using different sampling methods.

# In[10]:


from CreditCardApproval.model_training import split_data

X_train, X_val, y_train, y_val = split_data(X, y)


# In[11]:


from CreditCardApproval.model_training import train_glm, train_lgbm, run_training_all_sampling

# GLM: baseline + 3 sampling
glm_models_fixed, glm_train_auc_table = run_training_all_sampling(
    "glm", X_train, y_train, X_val, y_val
)

# LGBM: baseline + 3 sampling
lgbm_models_fixed, lgbm_train_auc_table = run_training_all_sampling(
    "lgbm", X_train, y_train, X_val, y_val
)

pd.concat([glm_train_auc_table, lgbm_train_auc_table], axis=0).reset_index(drop=True)


# ***

# ### 4. Hyperparameter Tuning

# The aim of this part is to find which sampling method has the highest AUC value, then choosing this method conducts hyperparameter tuning before moving into model evaluation.

# In[12]:


from CreditCardApproval.tuning import run_tuning_all_sampling

# GLM tuning: baseline + 3 sampling
glm_models_tuned, glm_tuned_auc_table = run_tuning_all_sampling(
    "glm", X_train, y_train, X_val, y_val, cv_splits=5
)

# LGBM tuning: baseline + 3 sampling
lgbm_models_tuned, lgbm_tuned_auc_table = run_tuning_all_sampling(
    "lgbm", X_train, y_train, X_val, y_val, cv_splits=5
)

pd.concat([glm_tuned_auc_table, lgbm_tuned_auc_table], axis=0).reset_index(drop=True)


# Review:
# 
# 【】The baseline LGBM without sampling achieves optimal validation AUC (0.764), outperforming all resampling strategies including tuned baseline (0.746). Hence, the baseline model with default hyperparameters is retained as the final configuration, demonstrating that aggressive hyperparameter optimization does not universally improve model performance and can occasionally introduce overfitting to the tuning process itself.

# ***

# ### 5. Model Evaluation & Comparison

# ##### ① Optimal Threshold

# In[13]:


from CreditCardApproval.evaluation import find_optimal_threshold

models_16 = {}

for s in ["baseline", "smote", "undersample", "combined"]:
    models_16[f"GLM_fixed_{s}"]  = glm_models_fixed[s]
    models_16[f"GLM_tuned_{s}"]  = glm_models_tuned[s]
    models_16[f"LGBM_fixed_{s}"] = lgbm_models_fixed[s]
    models_16[f"LGBM_tuned_{s}"] = lgbm_models_tuned[s]

threshold_rows = []
threshold_tables = {}

for name, model in models_16.items():
    y_proba = model.predict_proba(X_val)[:, 1]

    best_t, thr_table = find_optimal_threshold(
        y_true=y_val,
        y_proba=y_proba,
        metric="f2",
    )

    threshold_tables[name] = thr_table
    threshold_rows.append({
        "model_name": name,
        "optimal_threshold_f2": best_t,
        "best_f2": thr_table.loc[thr_table["threshold"].sub(best_t).abs().idxmin(), "f2"]
        if "f2" in thr_table.columns else None
    })

threshold_summary = (
    pd.DataFrame(threshold_rows)
      .sort_values(["model_name"])
      .reset_index(drop=True)
)

threshold_summary


# In[14]:


optimal_thresholds_16 = dict(
    zip(threshold_summary["model_name"], threshold_summary["optimal_threshold_f2"])
)
optimal_thresholds_16 = {k: optimal_thresholds_16[k] for k in models_16.keys()}


# In[15]:


from CreditCardApproval.evaluation import plot_threshold_curves_grid

plot_threshold_curves_grid(
    models=models_16, X=X_val, y=y_val,
    optimal_thresholds=optimal_thresholds_16,
    title="Threshold Curves (16 models, Validation)",
    save_name="16_Threshold_Curves",
    ncols=4
)


# Review: 
# 
# The optimal thresholds (GLM: 0.14, LGBM: 0.18) deviate substantially from the conventional 0.5 benchmark, directly reflecting the severe class imbalance (1-3% positive cases) in the dataset. LGBM demonstrates significant performance enhancement at the optimized threshold, with F2 scores improving from 0.09 to 0.32, which is a fourfold increase in detection capability.

# In[17]:


from CreditCardApproval.evaluation import evaluate_models_table

model_evaluation_outcomes = evaluate_models_table(
    models=models_16,
    X=X_val,
    y=y_val,
    thresholds=optimal_thresholds_16,
    sort_by="auc",
    ascending=False
)

model_evaluation_outcomes


# In[22]:


tmp = model_evaluation_outcomes["model_name"].str.split("_", expand=True)
model_evaluation_outcomes.insert(1, "algo", tmp[0])
model_evaluation_outcomes.insert(2, "params", tmp[1])
model_evaluation_outcomes.insert(3, "sampling", tmp[2])

model_evaluation_outcomes


# Review:
# 
# 1) LGBM substantially outperforms GLM across all configurations (AUC: 0.67-0.76 vs 0.43-0.56), confirming the inadequacy of linear models for this classification task.
# 2) LGBM baseline achieves the highest validation AUC (0.764) and demonstrates balanced performance metrics, suggesting that resampling techniques introduce noise rather than improvement for this dataset.
# 3) Undersampling and combined strategies yield extreme recall values (82-100%) at the cost of severely degraded precision (2-3%), indicating model collapse toward majority-class prediction. This pattern renders such configurations impractical for deployment.
# 4) LGBM baseline with optimized threshold (0.18) emerges as the recommended model, balancing discrimination capability (AUC 0.764) with operational metrics (recall 30%, precision 36%, F2 0.31).

# ***

# ### 6. Visualization

# ##### ① ROC Curve

# In[18]:


from CreditCardApproval.evaluation import plot_roc_curves_grid

plot_roc_curves_grid(
    models=models_16, X=X_val, y=y_val,
    title="ROC Curves (16 models, Validation)",
    save_name="16_ROC_Curves",
    ncols=4
)


# ##### ② Confusion Matrix

# In[19]:


from CreditCardApproval.evaluation import plot_confusion_matrices_grid

plot_confusion_matrices_grid(
    models=models_16, X=X_val, y=y_val,
    thresholds=optimal_thresholds_16,
    title="Confusion Matrices with Optimal Threshold (16 models, Validation)",
    save_name="16_Confusion_Matrices",
    ncols=4
)


# ##### ③ Compare predicted and actual data

# Based on previous model comparasion, a logistic regression model without resampling or hyperparameter tuning is adopted as a benchmark model due to its transparency and interpretability.
# 
# For the main risk assessment model, a LightGBM classifier trained on the original imbalanced dataset is selected. The model achieves the highest AUC and KS values among all candidates, indicating superior discriminatory power in ranking customer credit risk.
# 
# Notably, resampling techniques such as SMOTE and undersampling do not improve model performance, suggesting that preserving the original default distribution is critical for effective credit risk modeling in this dataset.

# In[27]:


from CreditCardApproval.evaluation import plot_predicted_vs_actual

best_glm_model = glm_models_fixed["baseline"]
best_lgbm_model = lgbm_models_fixed["baseline"]

final_models = {
    "Best_GLM": best_glm_model,
    "Best_LGBM": best_lgbm_model,
}

calib_table = plot_predicted_vs_actual(
    final_models,
    X_val,
    y_val,
    n_bins=10,
    strategy="quantile",
    title="Calibration: Predicted vs Actual (Test)",
    save_name="2_predicted_actual_calibration"
)

calib_table


# ##### ④ Feature importance

# In[30]:


from CreditCardApproval.evaluation import get_glm_feature_importance

glm_imp = get_glm_feature_importance(best_glm_model, top_k=10)
print(f"Feature importance of GLM:\n")
glm_imp


# In[31]:


from CreditCardApproval.evaluation import get_lgbm_feature_importance

lgbm_imp = get_lgbm_feature_importance(best_lgbm_model, top_k=10)
print(f"\nFeature importance of LGBM:\n")
lgbm_imp


# Review:
# 
# 

# ***

# In[ ]:




