import streamlit as st
import pickle
import numpy as np

# Load model
@st.cache_resource
def load_model():
    with open('best_random_forest.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Random Forest Prediction App")

st.write("Enter input values:")

# Example inputs (change based on your dataset)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    
    st.success(f"Prediction: {prediction[0]}")
