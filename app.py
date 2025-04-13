import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("churn_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎬 Customer Churn Prediction App")

# Input form
with st.form("churn_form"):
    year = st.number_input("Year", value=2025)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    no_days = st.slider("No. of Days Subscribed", 1, 365, 30)
    multi = st.selectbox("Multi-screen Access", ["Yes", "No"])
    mail = st.selectbox("Mail Subscribed", ["Yes", "No"])
    weekly_mins = st.number_input("Weekly Mins Watched", value=200)
    min_daily = st.number_input("Min Daily Mins", value=10)
    max_daily = st.number_input("Max Daily Mins", value=40)
    max_night = st.slider("Weekly Max Night Mins", 0, 300, 100)
    videos = st.slider("Videos Watched", 0, 100, 5)
    inactive_days = st.slider("Max Days Inactive", 0, 30, 3)
    calls = st.slider("Customer Support Calls", 0, 10, 0)
    submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([[
            year,
            0 if gender == "Male" else 1,
            age,
            no_days,
            1 if multi == "Yes" else 0,
            1 if mail == "Yes" else 0,
            weekly_mins,
            min_daily,
            max_daily,
            max_night,
            videos,
            inactive_days,
            calls
        ]], columns=[
            'year', 'gender', 'age', 'no_of_days_subscribed', 'multi_screen', 'mail_subscribed',
            'weekly_mins_watched', 'minimum_daily_mins', 'maximum_daily_mins',
            'weekly_max_night_mins', 'videos_watched', 'maximum_days_inactive', 'customer_support_calls'
        ])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("Result")
        st.success("✅ Will Not Churn" if prediction == 0 else "⚠️ Likely to Churn")
        st.info(f"Churn Probability: {prob:.2f}")
