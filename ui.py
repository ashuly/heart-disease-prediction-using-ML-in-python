import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore
import time

# Load model and preprocessing tools
model = load_model(r"C:\Users\DELL\Desktop\disease\heart_disease_model.h5")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("💓 Heart Disease Prediction App")
st.sidebar.header("🧾 Patient Information")

# Define custom inputs for certain fields
dropdowns = {
    "Gender": ["Male", "Female", "Other"],
    "Smoking": ["Yes", "No"],
    "Diabetes": ["Yes", "No"],
    "Family Heart Disease": ["Yes", "No"],
    "High Blood Pressure": ["Yes", "No"],
    "Low HDL Cholesterol": ["Yes", "No"],
    "High LDL Cholesterol": ["Yes", "No"]
}

yes_no_map = {"Yes": 1, "No": 0}
gender_map = {"Male": 0, "Female": 1, "Other": 2}

def get_user_input():
    user_data = {}
    for col in columns:
        if col in dropdowns:
            value = st.sidebar.selectbox(col, dropdowns[col])
            if value == "":
                st.error(f"Please select a value for {col}")
                return None
            user_data[col] = value
        elif col == "Age":
            user_data[col] = st.sidebar.number_input(col, min_value=0, max_value=120, step=1)
        elif "Level" in col or "Pressure" in col or "BMI" in col:
            user_data[col] = st.sidebar.slider(col, min_value=0.0, max_value=300.0, step=0.1)
        elif "Hours" in col:
            user_data[col] = st.sidebar.slider(col, min_value=0.0, max_value=24.0, step=0.5)
        else:
            user_data[col] = st.sidebar.number_input(col, value=0.0)

    return pd.DataFrame([user_data])

input_df = get_user_input()

# Add Predict Button
if st.button("🔮 Predict Heart Disease Risk"):
    if input_df is not None:
        # Encode categorical inputs
        for col in dropdowns:
            if col in input_df.columns:
                if col == "Gender":
                    input_df[col] = input_df[col].map(gender_map)
                else:
                    input_df[col] = input_df[col].map(yes_no_map)

        # Final check for missing values
        if input_df.isnull().any().any():
            st.error("❗ Some fields are missing or invalid.")
        else:
            st.subheader("🧾 Patient Input Data")
            st.write(input_df)

            # Simulate processing
            with st.spinner('🔄 Predicting... Please wait...'):
                time.sleep(1.5)  # Just to simulate progress for a smooth feel

                # Ensure correct column order
                input_df = input_df[columns]

                # Find numerical columns
                num_cols = input_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

                if not num_cols:
                    st.error("❗ No numerical columns found for scaling.")
                else:
                    # Scale the input
                    input_scaled = input_df.copy()
                    input_scaled[num_cols] = scaler.transform(input_df[num_cols])

                    # Predict
                    prediction_prob = model.predict(input_scaled.values)[0][0]
                    prediction = int(prediction_prob > 0.5)

                    # Fancy result
                    st.subheader("🩺 Prediction Result")
                    
                    if prediction == 1:
                        st.error(f"⚠️ High Risk of Heart Disease!\n\nConfidence: **{prediction_prob:.2%}**")
                    else:
                        st.success(f"✅ Low Risk of Heart Disease!\n\nConfidence: **{prediction_prob:.2%}**")

                    # Optional: nice progress bar to complete
                    progress_bar = st.progress(0)
                    for perc in range(100):
                        time.sleep(0.005)
                        progress_bar.progress(perc + 1)

else:
    st.info("👈 Fill in your details in the sidebar and click **Predict** to check risk!")

