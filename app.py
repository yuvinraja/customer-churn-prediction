
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("churn_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎬 Customer Churn Prediction App")

# Sample data definitions
sample_churn = {
    "year": 2025,
    "gender": "Female",
    "age": 35,
    "no_days": 25,
    "multi": "No",
    "mail": "No",
    "weekly_mins": 40,
    "min_daily": 5,
    "max_daily": 10,
    "max_night": 20,
    "videos": 1,
    "inactive_days": 10,
    "calls": 5
}

sample_no_churn = {
    "year": 2025,
    "gender": "Male",
    "age": 28,
    "no_days": 300,
    "multi": "Yes",
    "mail": "Yes",
    "weekly_mins": 300,
    "min_daily": 20,
    "max_daily": 50,
    "max_night": 150,
    "videos": 20,
    "inactive_days": 0,
    "calls": 0
}

# Sample data buttons
st.subheader("🧪 Test Sample Data")
sample_type = None
col1, col2 = st.columns(2)
with col1:
    if st.button("Will Churn (Sample)"):
        sample_type = sample_churn
with col2:
    if st.button("Will Not Churn (Sample)"):
        sample_type = sample_no_churn

# Input form
with st.form("churn_form"):
    year = st.number_input("Year", value=sample_type["year"] if sample_type else 2025)
    gender = st.selectbox("Gender", ["Male", "Female"], index=1 if sample_type and sample_type["gender"] == "Female" else 0)
    age = st.slider("Age", 18, 100, sample_type["age"] if sample_type else 30)
    no_days = st.slider("No. of Days Subscribed", 1, 365, sample_type["no_days"] if sample_type else 30)
    multi = st.selectbox("Multi-screen Access", ["Yes", "No"], index=0 if sample_type and sample_type["multi"] == "Yes" else 1)
    mail = st.selectbox("Mail Subscribed", ["Yes", "No"], index=0 if sample_type and sample_type["mail"] == "Yes" else 1)
    weekly_mins = st.number_input("Weekly Mins Watched", value=sample_type["weekly_mins"] if sample_type else 200)
    min_daily = st.number_input("Min Daily Mins", value=sample_type["min_daily"] if sample_type else 10)
    max_daily = st.number_input("Max Daily Mins", value=sample_type["max_daily"] if sample_type else 40)
    max_night = st.slider("Weekly Max Night Mins", 0, 300, sample_type["max_night"] if sample_type else 100)
    videos = st.slider("Videos Watched", 0, 100, sample_type["videos"] if sample_type else 5)
    inactive_days = st.slider("Max Days Inactive", 0, 30, sample_type["inactive_days"] if sample_type else 3)
    calls = st.slider("Customer Support Calls", 0, 10, sample_type["calls"] if sample_type else 0)
    submit = st.form_submit_button("Predict")

# Prediction block
if submit or sample_type:
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
