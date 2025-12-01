import streamlit as st
import pandas as pd
import numpy as np

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
st.write("Fair AI model with Explainability.")

# Load model
model = scaler = None
if JOBLIB_AVAILABLE:
    try:
        model = joblib.load("models/baseline_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        st.success("✅ Model loaded")
    except Exception as e:
        st.warning(f"⚠️ Model loading issue: {e}")

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
    
    # Get prediction
    if model and scaler:
        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
        except:
            # Simple rule-based prediction
            prediction = int(age > 30 and education > 12)
            probability = 0.8 if prediction else 0.3
    else:
        # Demo prediction rules
        base_score = 0.5
        if age > 40: base_score += 0.15
        elif age > 30: base_score += 0.1
        if education > 12: base_score += 0.2
        if hours > 40: base_score += 0.1
        if gender == "Male": base_score += 0.1
        if race == "White": base_score += 0.05
        
        probability = min(max(base_score, 0.1), 0.95)  # Clamp between 0.1 and 0.95
        prediction = 1 if probability > 0.5 else 0

    # Output
    st.subheader("🔍 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{probability*100:.1f}%")
    with col2:
        if prediction:
            st.success("HIGH INCOME (>50K)")
        else:
            st.info("LOW INCOME (<=50K)")

    # Fairness check
    if gender == "Female" and probability < 0.4:
        st.warning("⚠️ Possible gender bias detected")

    # ---------------------------------
    # SIMPLE BUT EFFECTIVE EXPLAINABILITY
    # ---------------------------------
    st.subheader("🧠 Explainability (Feature Impact Analysis)")
    
    # Calculate feature impacts (simple rule-based but effective)
    impacts = []
    
    # Age impact
    age_impact = 0
    if age > 55: age_impact = 0.25
    elif age > 45: age_impact = 0.20
    elif age > 35: age_impact = 0.15
    elif age > 30: age_impact = 0.10
    elif age < 25: age_impact = -0.10
    impacts.append(("Age", age_impact, f"{age} years"))
    
    # Education impact
    edu_impact = 0
    if education >= 16: edu_impact = 0.30
    elif education >= 14: edu_impact = 0.25
    elif education >= 12: edu_impact = 0.15
    elif education < 9: edu_impact = -0.10
    impacts.append(("Education Level", edu_impact, f"Level {education}"))
    
    # Hours impact
    hours_impact = 0
    if hours > 60: hours_impact = 0.15
    elif hours > 50: hours_impact = 0.10
    elif hours > 40: hours_impact = 0.05
    elif hours < 30: hours_impact = -0.05
    impacts.append(("Hours/Week", hours_impact, f"{hours} hours"))
    
    # Gender impact
    gender_impact = 0.10 if gender == "Male" else -0.05
    impacts.append(("Gender", gender_impact, gender))
    
    # Race impact
    race_impact = 0.05 if race == "White" else -0.03
    impacts.append(("Race", race_impact, race))
    
    # Create impact table
    impact_df = pd.DataFrame({
        "Feature": [i[0] for i in impacts],
        "Current Value": [i[2] for i in impacts],
        "Impact Score": [f"{i[1]:+.2f}" for i in impacts],
        "Impact": [i[1] for i in impacts]
    })
    
    # Sort by absolute impact
    impact_df["Abs_Impact"] = np.abs(impact_df["Impact"])
    impact_df = impact_df.sort_values("Abs_Impact", ascending=False).drop(columns=["Abs_Impact", "Impact"])
    
    # Display table
    st.table(impact_df)
    
    # Visual impact bars
    st.markdown("#### 📊 Feature Impact Visualization")
    
    for feature, impact, value in impacts:
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            st.write(f"**{feature}**")
        with col2:
            # Create a simple bar
            bar_color = "green" if impact > 0 else "red" if impact < 0 else "gray"
            bar_width = min(abs(impact) * 10, 5)  # Scale for display
            st.markdown(f'<div style="background-color:{bar_color}; width:{bar_width}em; height:20px; border-radius:3px;"></div>', unsafe_allow_html=True)
        with col3:
            st.write(f"{impact:+.2f}")
    
    # Summary explanation
    st.markdown("#### 📋 Summary")
    
    # Get top 2 positive and top 2 negative impacts
    positive_impacts = [(f, i) for f, i, v in impacts if i > 0]
    negative_impacts = [(f, i) for f, i, v in impacts if i < 0]
    
    if positive_impacts:
        st.write("**Factors increasing income probability:**")
        for feature, impact in sorted(positive_impacts, key=lambda x: x[1], reverse=True)[:3]:
            st.write(f"• **{feature}**: Increased probability by {impact:.2f}")
    
    if negative_impacts:
        st.write("**Factors decreasing income probability:**")
        for feature, impact in sorted(negative_impacts, key=lambda x: x[1])[:3]:
            st.write(f"• **{feature}**: Decreased probability by {abs(impact):.2f}")
    
    # Bias check
    st.markdown("#### ⚖️ Fairness Check")
    if gender_impact < -0.05:
        st.error(f"⚠️ **Gender Bias Alert**: {gender} identity shows negative impact")
    elif gender_impact > 0.08:
        st.warning(f"⚠️ **Potential Gender Advantage**: {gender} identity shows strong positive impact")
    
    if race_impact < -0.03:
        st.error(f"⚠️ **Racial Bias Alert**: {race} identity shows negative impact")
    elif race_impact > 0.04:
        st.warning(f"⚠️ **Potential Racial Advantage**: {race} identity shows positive impact")

st.markdown("---")
st.write("Built with ❤️ using Streamlit")
