import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Joblib import (optional model load)
# -------------------------------
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.warning("⚠️ joblib not available - using demo mode")

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="SHAP Explainability Demo", page_icon="🤖")
st.title("SHAP Feature Importance Demo")

# -------------------------------
# Demo data (replace with your dataset & model)
# -------------------------------
if JOBLIB_AVAILABLE:
    model = joblib.load("your_model.pkl")  # replace with your model path
    X = pd.read_csv("your_data.csv")       # replace with your data path
else:
    st.info("Demo mode: Generating synthetic data")
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor

    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)

# -------------------------------
# SHAP Explainer
# -------------------------------
st.subheader("SHAP Feature Importance")

try:
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # -------------------------------
    # Mean absolute SHAP values table
    # -------------------------------
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": mean_shap_values
    }).sort_values("Importance", key=abs, ascending=False)

    st.write("### Feature Importance Table")
    st.dataframe(shap_df)

    # -------------------------------
    # SHAP summary plot
    # -------------------------------
    st.write("### SHAP Summary Plot")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X, show=False)
    st.pyplot(plt.gcf())

except Exception as e:
    st.error(f"SHAP explanation failed: {e}")


st.markdown("---")
st.write("Built with ❤️ using Streamlit")
