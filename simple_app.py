import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="🏠 Boston Housing Predictor", page_icon="🏡")
st.title("Boston Housing Price Predictor with SHAP Explainability")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    boston = fetch_openml(name="Boston", version=1, as_frame=True)
    X = boston.data
    y = boston.target
    return X, y

X, y = load_data()
st.subheader("Dataset Preview")
st.dataframe(X.head())

# -------------------------------
# Train model
# -------------------------------
@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model(X, y)
st.success("✅ Model trained successfully!")

# -------------------------------
# SHAP explainability setup
# -------------------------------
@st.cache_resource
def calculate_shap(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Mean |SHAP value|": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="Mean |SHAP value|", ascending=False)
    return shap_df, explainer, shap_values

shap_df, explainer, shap_values = calculate_shap(model, X_train)

st.subheader("Global Feature Importance")
st.dataframe(shap_df)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_train, show=False)
st.pyplot(fig)

# -------------------------------
# Interactive prediction
# -------------------------------
st.subheader("Predict & Explain a Sample")

sample_index = st.slider("Select a sample row from training data", 0, X_train.shape[0]-1, 0)
sample = X_train.iloc[sample_index:sample_index+1]

prediction = model.predict(sample)[0]
st.write(f"Predicted Price for selected sample: **${prediction:.2f}k**")

# SHAP explanation for selected sample
st.write("SHAP Force Plot for selected sample:")
shap.initjs()
fig2, ax2 = plt.subplots(figsize=(10, 4))
shap.force_plot(explainer.expected_value, shap_values[sample_index,:], sample, matplotlib=True, show=False)
st.pyplot(fig2)

# Optionally display raw feature values
if st.checkbox("Show selected sample features"):
    st.write(sample)


st.markdown("---")
st.write("Built with ❤️ using Streamlit")
