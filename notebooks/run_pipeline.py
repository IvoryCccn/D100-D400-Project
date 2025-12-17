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

# In[10]:


from CreditCardApproval.model_training import split_data

X_train, X_val, y_train, y_val = split_data(X, y)


# In[11]:


from CreditCardApproval.model_training import train_glm

glm_model, glm_auc = train_glm(X_train, y_train, X_val, y_val)
glm_auc


# In[12]:


from CreditCardApproval.model_training import train_lgbm

lgbm_model, lgbm_auc = train_lgbm(X_train, y_train, X_val, y_val)
lgbm_auc


# ***

# ### 4. Hyperparameter Tuning

# In[13]:


from CreditCardApproval.tuning import tune_glm, tune_lgbm
from sklearn.metrics import roc_auc_score


# In[14]:


glm_best_model, glm_best_params = tune_glm(
    X_train,
    y_train,
    cv_splits=5,
)

glm_best_params


# In[15]:


glm_val_proba = glm_best_model.predict_proba(X_val)[:, 1]
glm_val_auc = roc_auc_score(y_val, glm_val_proba)

glm_val_auc


# In[16]:


lgbm_best_model, lgbm_best_params = tune_lgbm(
    X_train,
    y_train,
    cv_splits=5,
)

lgbm_best_params


# In[17]:


lgbm_val_proba = lgbm_best_model.predict_proba(X_val)[:, 1]
lgbm_val_auc = roc_auc_score(y_val, lgbm_val_proba)

lgbm_val_auc


# In[18]:


pd.DataFrame(
    {
        "Model": ["GLM (baseline)", "GLM (tuned)", "LGBM (baseline)", "LGBM (tuned)"],
        "Validation AUC": [
            glm_auc,
            glm_val_auc,
            lgbm_auc,
            lgbm_val_auc,
        ],
    }
)


# ***

# ### 5. Model Evaluation & Comparison

# In[19]:


from CreditCardApproval.evaluation import compare_models

summary = compare_models(
    models={
        "GLM (baseline)": glm_model,
        "GLM (tuned)": glm_best_model,
        "LGBM (baseline)": lgbm_model,
        "LGBM (tuned)": lgbm_best_model,
    },
    X=X_val,
    y=y_val,
    threshold=0.5,
)

summary


# ***

# ### 6. Visualization

# ##### ① ROC Curve

# In[20]:


from CreditCardApproval.evaluation import plot_roc_curves

plot_roc_curves(
    models={
        "GLM": glm_model,
        "LGBM": lgbm_model,
    },
    X=X_val,
    y=y_val,
)


# ##### ② Confusion Matrix

# In[21]:


from CreditCardApproval.evaluation import plot_confusion_matrix

plot_confusion_matrix(
    lgbm_model,
    X_val,
    y_val,
    threshold=0.3,
    title="LGBM Confusion Matrix (threshold=0.3)",
)


# ##### ③ Optimal Threshold

# In[22]:


from CreditCardApproval.evaluation import find_optimal_threshold

# 1) get predicted probabilities
glm_proba = glm_model.predict_proba(X_val)[:, 1]
lgbm_proba = lgbm_model.predict_proba(X_val)[:, 1]

# 2) find optimal thresholds
glm_best_t, glm_thr_table = find_optimal_threshold(
    y_true=y_val,
    y_proba=glm_proba,
    metric="f2",
)

lgbm_best_t, lgbm_thr_table = find_optimal_threshold(
    y_true=y_val,
    y_proba=lgbm_proba,
    metric="f2",
)

glm_best_t, lgbm_best_t


# In[23]:


from CreditCardApproval.evaluation import plot_threshold_curves

plot_threshold_curves(
    y_true=y_val,
    y_proba=glm_proba,
    title="GLM Threshold vs Metrics"
)


# In[24]:


from CreditCardApproval.evaluation import plot_threshold_curves

plot_threshold_curves(
    y_true=y_val,
    y_proba=lgbm_proba,
    title="LGBM Threshold vs Metrics"
)


# In[25]:


from CreditCardApproval.evaluation import compare_models

summary_opt = compare_models(
    models={
        "GLM (t=0.5)": glm_model,
        "GLM (opt F2)": glm_model,
        "LGBM (t=0.5)": lgbm_model,
        "LGBM (opt F2)": lgbm_model,
    },
    X=X_val,
    y=y_val,
    threshold={
        "GLM (t=0.5)": 0.5,
        "GLM (opt F2)": glm_best_t,
        "LGBM (t=0.5)": 0.5,
        "LGBM (opt F2)": lgbm_best_t,
    },
)

summary_opt


# ##### ④ Compare predicted and actual data

# In[26]:


from CreditCardApproval.evaluation import plot_predicted_vs_actual

cal_table = plot_predicted_vs_actual(
    models={"GLM": glm_model, "LGBM": lgbm_model},
    X=X_val,
    y=y_val,
    n_bins=10,
    strategy="quantile",
)

cal_table.head(10)


# ##### ⑤ Feature importance

# In[27]:


from CreditCardApproval.evaluation import (
    get_glm_feature_importance,
    get_lgbm_feature_importance,
)

glm_imp = get_glm_feature_importance(glm_model, top_k=10)
lgbm_imp = get_lgbm_feature_importance(lgbm_model, top_k=10)

print(f"Feature importance of GLM:\n", glm_imp)
print(f"\nFeature importance of LGBM:\n", lgbm_imp)


# ***

# ### 7. Conclusion

# 

# In[ ]:




