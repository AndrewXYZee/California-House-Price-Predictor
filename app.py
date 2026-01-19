#Load required libraries and model
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('cali.joblib')

#Setup website
st.title("California House Price Predictor")
st.markdown("Setup the parameters of the California House and find out its price!")
st.caption("Model RMSE = 0.5609 and R2 = 0.6528 on test set")

##Sidebars
med_inc = st.sidebar.slider("Median Income ($10k units)", 0.5, 15.0, 3.0)
house_age = st.sidebar.slider("House Age (years)", 1, 52, 20)
ave_rooms = st.sidebar.slider("Average Rooms", 1, 40, 6)
ave_bedrms = st.sidebar.slider("Avarage Bedrooms", 1, 12, 3)
pop = st.sidebar.slider("Population", 1000, 17000, 6000)
ave_occup = st.sidebar.slider("Average Occupancy", 1, 20, 5)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0)
longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -120.0)

rooms_per_person = ave_rooms / ave_occup

#Load input
input_data = pd.DataFrame({
    'MedInc': [med_inc], 'HouseAge': [house_age], 'AveRooms': [ave_rooms],
    'AveBedrms': [ave_bedrms], 'Population': [pop], 'AveOccup': [ave_occup],
    'Latitude': [latitude], 'Longitude': [longitude], 'log_MedInc': [np.log1p(med_inc)],
    'log_AveRooms': [np.log1p(ave_rooms)], 'log_AveBedrms': [np.log1p(ave_bedrms)],
    'log_AveOccup': [np.log1p(ave_occup)], 'rooms_per_person': [np.log1p(rooms_per_person)]
})

if st.button("Predict Price"):
    pred_price = model.predict(input_data)[0] * 100000  # Multiply to show in $
    st.success(f"Predicted Median Price: **${pred_price:,.0f}**")