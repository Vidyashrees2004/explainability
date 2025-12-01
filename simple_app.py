# simple_app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Joblib import (for loading model)
# -------------------------------
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.warning("⚠️ joblib not available — demo mode enabled")

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Fair AI Demo", page_icon="🤖")
st.title("Fair AI Demo with SHAP Explainability")

# -------------------------------
# Load data and model
# -------------------------------
if JOBLIB_AVAILABLE:
    model = joblib.load("model.joblib")  # Replace with your model path
    X_train = joblib.load("X_train.joblib")  # Replace with your training data
else:
    st.info("Demo mode: generating synthetic data")
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor

    data = load_boston()
    X_train = pd.DataFrame(data.data, columns=data.feature_names)
    y_train = pd.Series(data.target)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

# -------------------------------
# SHAP explainability setup
# -------------------------------
# Do NOT cache this function since SHAP objects are unhashable
def calculate_shap(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Mean |SHAP value|": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="Mean |SHAP value|", ascending=False)
    return shap_df, explainer, shap_values

shap_df, explainer, shap_values = calculate_shap(model, X_train)

# -------------------------------
# Display SHAP feature importance
# -------------------------------
st.subheader("SHAP Feature Importance")
st.dataframe(shap_df)

# -------------------------------
# SHAP summary plot
# -------------------------------
st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_train, show=False)
st.pyplot(fig)


st.markdown("---")
st.write("Built with ❤️ using Streamlit")
