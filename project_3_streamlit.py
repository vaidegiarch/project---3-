import streamlit as st
import pickle
import numpy as np

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
# Load Model
# -------------------------------
@st.cache_resource
def load_model():

    with open('/content/drive/MyDrive/1random_forest.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("🌲 Forest Cover Type Prediction")

st.write("Enter the values below to predict forest cover type")

# -------------------------------
# User Inputs (example features)
# -------------------------------
Elevation = st.number_input("Elevation", value=2500)
Aspect = st.number_input("Aspect", value=100)
Slope = st.number_input("Slope", value=10)
Horizontal_Distance_To_Hydrology = st.number_input("Distance to Hydrology", value=200)
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance to Hydrology", value=50)
Horizontal_Distance_To_Roadways = st.number_input("Distance to Roadways", value=500)
Hillshade_9am = st.number_input("Hillshade 9am", value=200)
Hillshade_Noon = st.number_input("Hillshade Noon", value=220)
Hillshade_3pm = st.number_input("Hillshade 3pm", value=150)
Horizontal_Distance_To_Fire_Points = st.number_input("Distance to Fire Points", value=1000)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):
    input_data = np.array([[Elevation, Aspect, Slope,
                            Horizontal_Distance_To_Hydrology,
                            Vertical_Distance_To_Hydrology,
                            Horizontal_Distance_To_Roadways,
                            Hillshade_9am, Hillshade_Noon,
                            Hillshade_3pm,
                            Horizontal_Distance_To_Fire_Points]])

    prediction = model.predict(input_data)[0]
    label = cover_type_labels.get(prediction, "Unknown")
    st.success(f"🌳Predicted Forest Cover Type: {label}")
