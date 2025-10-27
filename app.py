import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("house_price_model.pkl")

# Page config
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

# Title
st.title("🏠 House Price Prediction App")
st.write("Use this app to estimate the price of a house based on its features.")

# Sidebar for inputs
st.sidebar.header("📊 Enter House Details")

square_feet = st.sidebar.slider("Square Feet", 100, 10000, 1200, step=50)
num_rooms = st.sidebar.slider("Number of Rooms", 1, 15, 3)
age = st.sidebar.slider("Age of House (Years)", 0, 200, 10)
distance = st.sidebar.slider("Distance to City (km)", 0, 100, 5)

# Predict button
if st.sidebar.button("🔮 Predict Price"):
    input_data = np.array([[square_feet, num_rooms, age, distance]])
    prediction = model.predict(input_data)
    
    # Show result in a styled box
    st.markdown(
        f"""
        <div style="background-color:#f0f9ff;padding:20px;border-radius:15px;border:2px solid #1e3a8a;text-align:center;">
            <h2 style="color:#1e3a8a;">💰 Estimated Price</h2>
            <h1 style="color:#10b981;">${prediction[0]:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Extra info section
st.markdown("---")
st.subheader("ℹ️ About this App")
st.write("""
This tool uses a **machine learning model (Random Forest)** trained on housing data to predict property prices.  
Adjust the sliders in the sidebar and click **Predict Price** to get an instant estimate.
""")
