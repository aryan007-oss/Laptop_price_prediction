import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Load trained pipeline + metadata

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    meta = joblib.load("meta.pkl")
    return model, meta

model, meta = load_model()

st.title("💻 Laptop Price Predictor")
st.caption(
    f"Model: **{meta['best_model']}**  |  "
    f"Test R²: **{meta['test_r2']:.4f}**  |  "
    f"Test MAE: ₹**{meta['test_mae']:.2f}**"
)

st.write("Enter the laptop specifications below to estimate its price.")


# Input form
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", meta["brands"])
        processor_speed = st.slider(
            "Processor Speed (GHz)",
            float(meta["ranges"]["Processor_Speed"][0]),
            float(meta["ranges"]["Processor_Speed"][1]),
            float((meta["ranges"]["Processor_Speed"][0] + meta["ranges"]["Processor_Speed"][1]) / 2),
        )
        ram_size = st.selectbox("RAM Size (GB)", [4, 8, 16, 32])
        storage_capacity = st.selectbox("Storage Capacity (GB)", [128, 256, 512, 1000])

    with col2:
        screen_size = st.slider(
            "Screen Size (inches)",
            float(meta["ranges"]["Screen_Size"][0]),
            float(meta["ranges"]["Screen_Size"][1]),
            float((meta["ranges"]["Screen_Size"][0] + meta["ranges"]["Screen_Size"][1]) / 2),
        )
        weight = st.slider(
            "Weight (kg)",
            float(meta["ranges"]["Weight"][0]),
            float(meta["ranges"]["Weight"][1]),
            float((meta["ranges"]["Weight"][0] + meta["ranges"]["Weight"][1]) / 2),
        )

    submitted = st.form_submit_button("Predict Price")

# Predict

if submitted:
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Processor_Speed": processor_speed,
        "RAM_Size": ram_size,
        "Storage_Capacity": storage_capacity,
        "Screen_Size": screen_size,
        "Weight": weight,
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"### Estimated Price: ₹{prediction:,.2f}")
    st.write("Input summary:")
    st.dataframe(input_df, use_container_width=True)
