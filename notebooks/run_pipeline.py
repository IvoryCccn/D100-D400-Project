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


# Review:
# 
# Results indicate that:
# 1) The LGBM model demonstrated significantly superior overall discrimination capability compared to GLM and The LGBM baseline model achieved the highest validation set AUC.
# 2) GLM generally performed poorly across different sampling methods, with its discrimination capability further declining after applying SMOTE or undersampling. This suggests that linear models have limited adaptability to resampling methods when dealing with complex feature structures and highly imbalanced categories, making it difficult to fully identify the distinctive features of high-risk customers.
# 3) For the LGBM model, although resampling methods partially improved the class structure, the validation set AUC remained below the baseline model. This indicates that the original sample distribution already provided sufficiently effective risk signals for the model on this dataset. Over-adjusting the sample structure may weaken the model's ability to learn from the true risk distribution.

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
# After tuning, the LGBM model demonstrates consistent overall performance across different sampling strategies. Results indicate that:
# 1) LGBM SMOTE model achieves the highest cross-validation AUC, indicating strong in-sample performance under resampled training data. But its validation AUC is noticeably lower, suggesting that the improvement observed during cross-validation does not fully generalize to unseen data.
# 2) LGBM baseline model attains the highest validation AUC with a much smaller gap between cross-validation and validation performance, reflecting better model stability and generalization.

# ***

# ### 5. Model Evaluation & Comparison

# ##### ① Optimal Threshold

# In[16]:


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


# In[17]:


optimal_thresholds_16 = dict(
    zip(threshold_summary["model_name"], threshold_summary["optimal_threshold_f2"])
)
optimal_thresholds_16 = {k: optimal_thresholds_16[k] for k in models_16.keys()}


# In[18]:


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
# Threshold curves illustrate how key performance metrics, including precision, recall, F1, and F2, vary as the classification threshold changes. It highlights the trade-offs between identifying high-risk applicants and controlling false positives, and provide a practical basis for selecting decision thresholds aligned with credit risk management objectives.
# 
# Compared to GLM, the LGBM model maintains relatively stable F2 performance across a broader threshold range, demonstrating greater flexibility in risk identification. This suggests that in credit approval scenarios, threshold adjustments can effectively balance default detection capability with misclassification costs, thereby enhancing the model's practical application value.

# In[19]:


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


# In[20]:


tmp = model_evaluation_outcomes["model_name"].str.split("_", expand=True)
model_evaluation_outcomes.insert(1, "algo", tmp[0])
model_evaluation_outcomes.insert(2, "params", tmp[1])
model_evaluation_outcomes.insert(3, "sampling", tmp[2])

model_evaluation_outcomes


# Review:
# 
# The meaning of each evaluation factor is:
# * threshold: The probability cutoff used to classify applicants as default or non-default.
# * AUC: Measures the model’s overall ability to rank higher-risk applicants above lower-risk ones.
# * gini: A scaled version of AUC commonly used in credit risk to quantify discriminatory power.
# * KS: Captures the maximum separation between good and bad customers across score distributions.
# * accuracy: The proportion of correctly classified applicants across all observations.
# * precision: The share of predicted defaulters that are truly default cases.
# * recall: The proportion of actual defaulters correctly identified by the model.
# * f1: The harmonic mean of precision and recall, balancing false positives and false negatives.
# * f2: A recall-weighted metric that places greater emphasis on identifying defaulters.
# * tp: Number of defaulters correctly identified as high risk.
# * fp: Number of non-defaulters incorrectly classified as defaulters.
# * tn: Number of non-defaulters correctly classified as low risk.
# * fn: Number of defaulters incorrectly classified as non-defaulters.
# 
# Results indicate that:
# 1) Across all configurations, LGBM demonstrated significantly superior discrimination capabilities compared to GLM, achieving markedly higher validation set AUC values. This indicates that linear models struggle to effectively capture the complex credit risk structure within this dataset.
# 2) LGBM achieved the highest validation set AUC under the baseline scenario without resampling, exhibiting relatively balanced performance across metrics. This suggests that resampling methods did not yield stable improvements on this dataset and may instead introduce noise.
# 3) While undersampling and hybrid sampling significantly improved recall, they came at the cost of substantial accuracy decline. This resulted in excessively high misclassification costs, diminishing the model's practicality for real-world credit approval.
# 4) Balancing discriminative capability with business metric performance, the LGBM tuned baseline model with optimized thresholds achieved an optimal equilibrium between risk identification effectiveness and operational feasibility. It is therefore recommended as the final model.

# ***

# ### 6. Visualization

# ##### ① ROC Curve

# In[21]:


from CreditCardApproval.evaluation import plot_roc_curves_grid

plot_roc_curves_grid(
    models=models_16, X=X_val, y=y_val,
    title="ROC Curves (16 models, Validation)",
    save_name="16_ROC_Curves",
    ncols=4
)


# Review:
# 
# The ROC curve characterizes the trade-off between true positive rate and false positive rate at different decision thresholds. Its area under the curve (AUC) reflects the model's overall ability to distinguish high-quality from low-quality customers, independent of specific threshold settings.
# 
# Results indicate that:
# 1) The ROC curve for the LGBM model consistently outperforms GLM, with a significantly larger AUC, demonstrating superior discrimination capabilities across varying risk preferences.
# 2) Consistent with prior findings, the LGBM model trained on the original sample distribution and optimized through parameter tuning exhibits optimal stability and discriminative power.

# ##### ② Confusion Matrix

# In[22]:


from CreditCardApproval.evaluation import plot_confusion_matrices_grid

plot_confusion_matrices_grid(
    models=models_16, X=X_val, y=y_val,
    thresholds=optimal_thresholds_16,
    title="Confusion Matrices with Optimal Threshold (16 models, Validation)",
    save_name="16_Confusion_Matrices",
    ncols=4
)


# Review:
# 
# Confusion matrices provide a visual representation of a model's classification results for high-quality versus low-quality customers at a given decision threshold. They reflect the number of defaults identified, the rate of false rejections of high-quality customers, and potential risk exposure, serving as a crucial basis for evaluating a model's actual risk control effectiveness.
# 
# Aligning with prior findings, LGBM tuned baseline maintains strong default detection capability while effectively controlling misclassification rates among high-quality customers, demonstrating balanced risk trade-offs. In contrast, some resampling models identify more high-risk customers but generate substantial misclassifications, increasing unnecessary rejection costs and limiting their practical applicability in credit approval workflows.

# ##### ③ Compare predicted and actual data

# Based on previous model comparasion, a baseline model with hyperparameter tuning is adopted as a benchmark model for both GLM and LGBM.

# In[24]:


from CreditCardApproval.evaluation import plot_predicted_vs_actual

best_glm_model = glm_models_tuned["baseline"]
best_lgbm_model = lgbm_models_tuned["baseline"]

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
    title="Calibration: Predicted vs Actual",
    save_name="2_predicted_actual_calibration"
)

calib_table


# Review:
# 
# The calibration curve assesses the consistency between the model's predicted default probabilities and actual occurrence rates. The closer the curve aligns with the diagonal line, the more accurately the model's risk probabilities reflect the customer's credit quality.
# 
# Results indicate that:
# 1) GLM's predicted probabilities closely match actual default rates across all bins, demonstrating overall good calibration. This highlights the stability of linear models in probability interpretation.
# 2) LGBM significantly underestimates default probabilities in low-risk intervals while exhibiting a more concentrated probability distribution in high-risk bins, suggesting its greater emphasis on risk ranking rather than probability characterization.
# 
# GLM demonstrates greater robustness in probabilistic characterization and stability, while LGBM exhibits distinct advantages in credit risk ranking and discrimination capabilities. Within risk control systems, the two models play complementary roles. Hence,despite LGBM's superior discriminative capability, practical risk control applications still necessitate calibration to enhance the interpretability and operational usability of its probability outputs.

# ##### ④ Feature importance

# In[25]:


from CreditCardApproval.evaluation import get_glm_feature_importance

glm_imp = get_glm_feature_importance(best_glm_model, top_k=10)
print(f"Feature importance of GLM:\n")
glm_imp


# In[26]:


from CreditCardApproval.evaluation import get_lgbm_feature_importance

lgbm_imp = get_lgbm_feature_importance(best_lgbm_model, top_k=10)
print(f"\nFeature importance of LGBM:\n")
lgbm_imp


# Review:
# 
# Feature importance analysis identifies the primary information sources relied upon by the model during credit risk assessment. In this context, GLM coefficients reflect the linear direction and strength of a variable's impact on default probability, while LGBM importance measures a feature's contribution to risk differentiation during tree model splitting.
# 
# Results indicate that: 
# 1) GLM relies more heavily on asset and social attribute variables—such as property/vehicle ownership, employment tenure, and family/marital status—demonstrating its linear modeling capability for customer stability and long-term credit characteristics.
# 2) LGBM emphasizes continuous variables and nonlinear relationships, with age, employment tenure, and income level playing dominant roles in risk differentiation, highlighting its advantage in capturing complex interaction effects.
# 
# The differing emphasis on features between the two models further validates their complementary roles in credit risk assessment—one emphasizing interpretability and the other emphasizing discrimination capability.

# In[ ]:




