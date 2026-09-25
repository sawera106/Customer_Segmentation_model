import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
scaler = jb.load('scaler.pkl')
model = jb.load('customer_segmentation_model.pkl')

st.title("AI Customer Segmentation Prediction")
st.warning("This is a simple AI model that predicts customer segments based on Annual Income and Spending Score. Please enter the required information below to get the predicted segment.")
annual_income = st.number_input("Enter Annual Income (in k):", min_value=10, max_value=200, value=50)
spending_score = st.number_input("Enter Spending Score :", min_value=1, max_value=200, value=50)
if st.button("Predict Segment"):
    # Preprocess the input data
    input_data = np.array([[annual_income, spending_score]])
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    predicted_segment = model.predict(input_data_scaled)
    
    # Display the result
    st.write(f"Predicted Customer Segment: {predicted_segment[0]}")
profiles = {
    0: 'VIPs - High income and high spending score',
    1: 'Savers - High income and low spending score',
    2: 'Spenders - Low income and high spending score'
}
# st.success(f"Customer Profile: {profiles[predicted_segment[0]]}")