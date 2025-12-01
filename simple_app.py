import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Joblib import
# -------------------------------
try:
    import joblib
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False
    st.warning("⚠️ joblib missing — demo mode enabled")

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Fair AI Demo", page_icon="🤖")
st.title("🎯 Fair AI Income Predictor")
st.write("Predict income with Fair AI and explain predictions using SHAP.")

# ------------------------------
