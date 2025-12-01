import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="SHAP Explainability", page_icon="🤖")
st.title("SHAP Feature Importance Demo")

# -------------------------------
# Load demo dataset & model
# -------------------------------
st.info("Using demo Boston dataset with RandomForest model")
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# -------------------------------
# SHAP Explainer
# -------------------------------
st.subheader("SHAP Feature Importance Table and Plot")

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# -------------------------------
# Mean absolute SHAP values table
# -------------------------------
mean_shap_values = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": mean_shap_values
}).sort_values("Importance", ascending=False)

st.write("### Feature Importance Table")
st.dataframe(shap_df)

# -------------------------------
# SHAP summary plot
# -------------------------------
st.write("### SHAP Summary Plot")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values.values, X, show=False)
st.pyplot(plt.gcf())
plt.clf()


st.markdown("---")
st.write("Built with ❤️ using Streamlit")
