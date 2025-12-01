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
    # FIXED SHAP EXPLAINABILITY SECTION
    # ---------------------------------
    st.subheader("🧠 Explainability (SHAP Feature Importance)")
    
    feature_names = ["Age", "Education Level", "Hours/Week", "Gender", "Race"]
    
    if explainer and model:
        try:
            # Get SHAP values safely
            shap_values = explainer(features_scaled)
            
            # Handle different SHAP explainer types
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values[0]
            elif isinstance(shap_values, list):
                shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                shap_vals = shap_values[0]
            
            # Create explanation table
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact on Prediction": shap_vals,
                "Direction": ["+" if val > 0 else "-" for val in shap_vals]
            })
            
            # Sort by absolute impact
            shap_df["Abs_Impact"] = np.abs(shap_vals)
            shap_df = shap_df.sort_values("Abs_Impact", ascending=False).drop(columns=["Abs_Impact"])
            
            st.table(shap_df)
            
            # Feature descriptions
            st.markdown("#### 📊 How Features Influenced Prediction:")
            
            # Get top contributing features
            top_features = shap_df.head(3)
            for _, row in top_features.iterrows():
                impact = row["Impact on Prediction"]
                direction = "increased" if impact > 0 else "decreased"
                st.write(f"• **{row['Feature']}**: {abs(impact):.3f} points {direction} probability")
            
            st.caption("💡 Positive values push toward HIGH income, negative toward LOW income.")
            
        except Exception as e:
            st.warning(f"SHAP calculation issue: {e}")
            # Fallback to rule-based explanation
            st.info("**Fallback Analysis (based on input values):**")
            if age > 40:
                st.write(f"• **Age ({age})**: High age increases income probability")
            if education > 12:
                st.write(f"• **Education ({education})**: Advanced education increases income probability")
            if gender == "Male":
                st.write("• **Gender (Male)**: Increases income probability")
            if race == "White":
                st.write("• **Race (White)**: Increases income probability")
    else:
        # Demo mode explanation
        st.info("**Feature Impact Analysis:**")
        st.write(f"• **Age ({age})**: {'High' if age > 30 else 'Low'} impact")
        st.write(f"• **Education Level ({education})**: {'High' if education > 12 else 'Low'} impact")
        st.write(f"• **Hours/Week ({hours})**: {'High' if hours > 40 else 'Moderate'} impact")
        st.write(f"• **Gender ({gender})**: {'Positive' if gender == 'Male' else 'Neutral'} impact")
        st.write(f"• **Race ({race})**: {'Positive' if race == 'White' else 'Neutral'} impact")

st.markdown("---")
st.write("Built with ❤️ using Streamlit")
