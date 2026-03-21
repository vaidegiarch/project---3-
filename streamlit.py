import streamlit as st
import requests
import gzip
import pickle

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?export=download&id=1TfblkvNkWo08Tfite8BltolStRIXgWVU"
    
    response = requests.get(url)

    with open("model.pkl.gz", "wb") as f:
        f.write(response.content)

    with gzip.open("model.pkl.gz", "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🌳 Random Forest Prediction App")
st.write("Enter input values below:")

# Input fields (modify based on your dataset)
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
            st.error(f"❌ Prediction error: {type(e).__name__} - {e}")
    else:
        st.warning("⚠️ Model not loaded")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit 🚀")
