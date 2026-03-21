import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Load Model (with debug)
# -------------------------------
@st.cache_resource
def load_model():
    file_path = "best_random_forest.pkl"
    
    st.write("📁 Files in current directory:", os.listdir())

    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"❌ Model file NOT FOUND: {file_path}")
        return None

model = load_model()


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Random Forest App", layout="centered")

st.title("🌳 Random Forest Prediction App")
st.write("Enter input values below:")

# Input fields (modify as per your dataset)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if model is not None:
        try:
            input_data = np.array([[feature1, feature2, feature3]])
            prediction = model.predict(input_data)
            st.success(f"✅ Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.warning("⚠️ Model is not loaded.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit 🚀")
