import streamlit as st
import numpy as np
import joblib

# Load the trained model safely
try:
    model = joblib.load("crop_model.joblib")   # your trained model
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and weather conditions to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Recommend Crop"):
    if model:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        st.success(f"✅ Recommended Crop: **{prediction[0]}**")
    else:
        st.warning("⚠️ Model not loaded. Please check the file and scikit-learn version.")
