import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please check the path.")
        return None

model = load_model()

# -------------------------------
# Label Mapping
# -------------------------------
cover_type_labels = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# -------------------------------
# App Title
# -------------------------------
st.title("🌲 Forest Cover Type Prediction")

st.write("Enter the details below to predict the forest cover type.")

# -------------------------------
# Input Fields (example features)
# -------------------------------
Elevation = st.number_input("Elevation", min_value=0)
Aspect = st.number_input("Aspect", min_value=0)
Slope = st.number_input("Slope", min_value=0)
Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance To Hydrology", min_value=0)
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance To Hydrology")
Horizontal_Distance_To_Roadways = st.number_input("Horizontal Distance To Roadways", min_value=0)
Hillshade_9am = st.number_input("Hillshade 9am", min_value=0, max_value=255)
Hillshade_Noon = st.number_input("Hillshade Noon", min_value=0, max_value=255)
Hillshade_3pm = st.number_input("Hillshade 3pm", min_value=0, max_value=255)
Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal Distance To Fire Points", min_value=0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):
    if model is not None:
        try:
            input_data = np.array([[Elevation, Aspect, Slope,
                                    Horizontal_Distance_To_Hydrology,
                                    Vertical_Distance_To_Hydrology,
                                    Horizontal_Distance_To_Roadways,
                                    Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
                                    Horizontal_Distance_To_Fire_Points]])

            prediction = model.predict(input_data)[0]

            label = cover_type_labels.get(prediction, "Unknown")

            st.success(f"🌳 Predicted Forest Cover Type: {label}")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Model is not loaded properly.")
