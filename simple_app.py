import streamlit as st
import pandas as pd
import numpy as np
import shap


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

# -------------------------------
# Load models
# -------------------------------
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

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Details")
age = st.slider("Age", 18, 80, 35)
education = st.slider("Education Level (years)", 1, 16, 13)
hours = st.slider("Hours per Week", 10, 80, 40)
gender = st.selectbox("Gender", ["Female", "Male"])
race = st.selectbox("Race", ["Non-White", "White"])

gender_num = 1 if gender == "Male" else 0
race_num = 1 if race == "White" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Income"):
    features = np.array([[age, education, hours, gender_num, race_num]])

    if model and scaler:
        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            prediction = int(age > 30 and education > 12)
            probability = 0.8 if prediction else 0.3
    else:
        features_scaled = features  # demo mode
        prediction = int(age > 30 and education > 12)
        probability = 0.8 if prediction else 0.3

    # -------------------------------
    # Output Prediction
    # -------------------------------
    st.subheader("🔍 Prediction Result")
    st.metric("Confidence", f"{probability:.1%}")
    if prediction:
        st.success("HIGH INCOME (>50K)")
    else:
        st.info("LOW INCOME (<=50K)")

    # Fairness check
    if gender == "Female" and probability < 0.4:
        st.warning("⚠️ Possible gender bias detected")

    # -------------------------------
    # SHAP Explainability
    # -------------------------------
    st.subheader("🧠 Explainability (SHAP)")
    try:
        if model is not None and JOBLIB_AVAILABLE:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_scaled)

            # Binary classifier: pick positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Single-row prediction
            if shap_values.shape[0] == 1:
                shap_values = shap_values[0]

            feature_names = ["Age", "Education", "Hours", "Gender", "Race"]

            # Table
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": shap_values
            }).sort_values("Importance", key=abs, ascending=False)
            st.write("### 🔍 Feature Importance Table")
            st.dataframe(shap_df)

           
        else:
            st.warning("Explainability not available in demo mode.")
    except Exception as e:
        st.error(f"Explainability unavailable: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

st.markdown("---")
st.write("Built with ❤️ using Streamlit")
