import streamlit as st
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("subterra_decision_tree.pkl")

model = load_model()

# Define feature input names
feature_names = [
    "elevation", "slope", "tri", "fault_distance", "ndvi",
    "landcover", "veg_peak", "lithology", "heat_flow", "lst"
]

st.title("🌋 SubTerra: Geothermal Site Classifier")

st.markdown("Enter the feature values below:")

# Collect user inputs
inputs = []
for name in feature_names:
    value = st.number_input(f"{name}", step=0.01)
    inputs.append(value)

# Predict button
if st.button("Predict Geothermal Potential"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    if prediction == 1:
        st.success("✅ This site has high geothermal potential.")
    else:
        st.warning("⚠️ This site has low geothermal potential.")
