import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("Heart_Disease_Pred/heart_disease_model.pkl")
scaler = joblib.load("Heart_Disease_Pred/scaler.pkl")

st.title("💓 Heart Disease Prediction App")

# Sidebar input
st.sidebar.header("🧾 Patient Medical Info")

def user_input():
    age = st.sidebar.slider(
        "Age", 20, 80, 50,
        help="Age of the patient in years"
    )
    sex = st.sidebar.selectbox(
        "Sex", [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="0 = Female, 1 = Male"
    )
    chest_pain = st.sidebar.slider(
        "Chest Pain Type (1–4)", 1, 4, 2,
        help="1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic"
    )
    resting_bp = st.sidebar.slider(
        "Resting Blood Pressure (mm Hg)", 80, 200, 120,
        help="Resting systolic blood pressure in mm Hg"
    )
    cholesterol = st.sidebar.slider(
        "Cholesterol (mg/dl)", 100, 600, 200,
        help="Serum cholesterol level in mg/dl"
    )
    fasting_sugar = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl", [0, 1],
        help="1 = True, 0 = False"
    )
    resting_ecg = st.sidebar.slider(
        "Resting ECG (0–2)", 0, 2, 1,
        help="0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy"
    )
    max_hr = st.sidebar.slider(
        "Maximum Heart Rate Achieved", 60, 210, 150,
        help="Max heart rate achieved during exercise"
    )
    exercise_angina = st.sidebar.selectbox(
        "Exercise Induced Angina", [0, 1],
        help="1 = Yes, 0 = No"
    )
    oldpeak = st.sidebar.slider(
        "Oldpeak", -2.0, 6.0, 1.0,
        help="ST depression induced by exercise (relative to rest)"
    )
    st_slope = st.sidebar.slider(
        "ST Slope (0–3)", 0, 3, 1,
        help="0 = Upsloping, 1 = Flat, 2 = Downsloping"
    )

    data = {
        "age": age,
        "sex": sex,
        "chest pain type": chest_pain,
        "resting bp s": resting_bp,
        "cholesterol": cholesterol,
        "fasting blood sugar": fasting_sugar,
        "resting ecg": resting_ecg,
        "max heart rate": max_hr,
        "exercise angina": exercise_angina,
        "oldpeak": oldpeak,
        "ST slope": st_slope,
    }
    return pd.DataFrame([data])


input_df = user_input()

# Show user input
st.subheader("📝 Input Data")
st.write(input_df)

# Prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][prediction]

# Output
st.subheader("🧠 Prediction Result")
st.write(f"🔮 The patient is **{'at risk' if prediction == 1 else 'not at risk'}** of heart disease.")
st.write(f"🧪 Model confidence: **{proba:.2f}**")

# Optional: Match with known result
known = st.checkbox("Compare with real result?")
if known:
    real = st.selectbox("Actual Diagnosis", [0, 1], format_func=lambda x: "No Heart Disease" if x == 0 else "Has Heart Disease")
    if real == prediction:
        st.success("✅ Prediction matches the real result!")
    else:
        st.error("❌ Prediction does NOT match the real result.")
