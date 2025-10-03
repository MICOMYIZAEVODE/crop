import streamlit as st
import pandas as pd
import joblib
import os

# --- Page config ---
st.set_page_config(page_title="🌾 Crop Recommendation", layout="centered")
st.title("🌾 Crop Recommendation System")

# --- Load model safely ---
@st.cache_resource
def load_model(model_path="crop_model.joblib"):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please retrain and upload it.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# --- Load dataset safely (CSV preferred) ---
@st.cache_data
def load_data():
    if os.path.exists("Crop_recommendation.csv"):
        try:
            return pd.read_csv("Crop_recommendation.csv")
        except Exception as e:
            st.error(f"❌ Error reading CSV dataset: {e}")
            return None
    elif os.path.exists("Crop_recommendation.xlsx"):
        try:
            return pd.read_excel("Crop_recommendation.xlsx")
        except Exception as e:
            st.error(f"❌ Error reading Excel dataset: {e}")
            return None
    else:
        st.warning("⚠️ Dataset file not found. Upload CSV or Excel to enable predictions.")
        return None

# --- Load resources ---
model = load_model()
data = load_data()

# --- Check if everything is ready ---
if model is not None and data is not None:
    st.success("✅ Model and dataset loaded successfully!")

    # --- Input fields for prediction ---
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, step=1)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, step=1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

    # --- Predict button ---
    if st.button("🌱 Recommend Crop"):
        try:
            features = [[N, P, K, temperature, humidity, ph, rainfall]]
            prediction = model.predict(features)
            st.success(f"🌾 Recommended Crop: **{prediction[0]}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

else:
    st.info("📂 Please ensure the model and dataset are uploaded to use this app.")
