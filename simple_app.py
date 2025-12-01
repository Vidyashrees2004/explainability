import streamlit as st
import pandas as pd
import numpy as np
import shap

# Import joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False
    st.warning("⚠️ joblib missing — demo mode enabled")

# Streamlit settings
st.set_page_config(page_title="Fair AI Demo", page_icon="🤖")
st.title("🎯 Fair AI Income Predictor")
st.write("Fair AI model with Explainability (SHAP).")

# Load models
model = scaler = explainer = None

if JOBLIB_AVAILABLE:
    try:
        model = joblib.load("models/baseline_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        explainer = joblib.load("models/explainer.pkl")
        st.success("✅ Models + Explainer loaded")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
else:
    st.warning("⚠️ joblib not installed — running demo rules.")

# Inputs
st.subheader("Enter Details")
age = st.slider("Age", 18, 80, 35)
education = st.slider("Education Level", 1, 16, 13)
hours = st.slider("Hours per Week", 10, 80, 40)
gender = st.selectbox("Gender", ["Female", "Male"])
race = st.selectbox("Race", ["Non-White", "White"])

gender_num = 1 if gender == "Male" else 0
race_num = 1 if race == "White" else 0

if st.button("Predict Income"):
    features = np.array([[age, education, hours, gender_num, race_num]])

    if model:
        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
        except:
            prediction = int(age > 30 and education > 12)
            probability = 0.8 if prediction else 0.3
    else:
        prediction = int(age > 30 and education > 12)
        probability = 0.8 if prediction else 0.3

    # Output
    st.subheader("🔍 Prediction Result")
    st.metric("Confidence", f"{probability:.1%}")
    if prediction:
        st.success("HIGH INCOME (>50K)")
    else:
        st.info("LOW INCOME (<=50K)")

    # Fairness check
    if gender == "Female" and probability < 0.4:
        st.warning("⚠️ Possible gender bias detected")

    # ---------------------------------
    # SHAP EXPLAINABILITY SECTION
    # ---------------------------------
    st.subheader("🧠 Explainability (SHAP Feature Importance)")

    if explainer:
        try:
            shap_values = explainer.shap_values(features_scaled)[1]

            shap_df = pd.DataFrame({
                "Feature": ["Age", "Education", "Hours", "Gender", "Race"],
                "SHAP Value": shap_values[0]
            })

            st.table(shap_df)
            st.caption("Higher SHAP values → stronger contribution to HIGH income.")
        except Exception as e:
            st.error(f"SHAP Error: {e}")
    else:
        st.info("SHAP explainer not available.")

st.markdown("---")
st.write("Built with ❤️ using Streamlit")
