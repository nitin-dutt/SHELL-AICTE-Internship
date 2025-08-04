import streamlit as st
import numpy as np
import joblib


model = joblib.load("Farm_Irrigation_System.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Smart Sprinkler System", layout="centered")

st.title("Smart Irrigation Prediction System")
st.markdown("Predict which **parcel sprinklers** should turn ON based on your sensor data.")

st.markdown("### Enter scaled sensor readings (0.00 - 1.00):")

sensor_values = []
cols = st.columns(3)

for i in range(20):
    with cols[i % 3]:
        val = st.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        sensor_values.append(val)

if st.button("🔍 Predict Sprinkler Status"):
    input_array = np.array(sensor_values).reshape(1, -1)

    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)[0]

    st.success("✅ Prediction Complete!")
    st.markdown("### 💡 Sprinkler Decisions:")

    for i, status in enumerate(prediction):
        color = "green" if status == 1 else "red"
        state = "ON" if status == 1 else "OFF"
        st.markdown(
            f"<span style='background-color:{color}; color:white; padding:5px 15px; border-radius:10px;'>"
            f"Sprinkler {i} (parcel_{i}): {state}</span>",
            unsafe_allow_html=True
        )
