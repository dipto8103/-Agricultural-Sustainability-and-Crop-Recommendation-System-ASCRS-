################################################################################
################################################################################


import streamlit as st
import numpy as np
import pickle

pickle_rain_model = open("/Users/shreyassawant/mydrive/Shreyus_workspace/IITB_hackathon_code/trained_model.pkl", "rb")
rain_model = pickle.load(pickle_rain_model)
pickle_rain_scaler =  open("/Users/shreyassawant/mydrive/Shreyus_workspace/IITB_hackathon_code/rain_scaler.pkl", "rb")
rain_scaler = pickle.load(pickle_rain_scaler)

pickle_in_lr = open("/Users/shreyassawant/mydrive/Shreyus_workspace/IITB_hackathon_code/rainfall_model.pkl", "rb")
lr_model = pickle.load(pickle_in_lr)
pickle_in_encoder = open("/Users/shreyassawant/mydrive/Shreyus_workspace/IITB_hackathon_code/label_encoder.pkl", "rb")
scaler = pickle.load(pickle_in_encoder)
pickle_in_scaler = open("/Users/shreyassawant/mydrive/Shreyus_workspace/IITB_hackathon_code/scaler.pkl", "rb")
scaler = pickle.load(pickle_in_scaler)

#test fuction
def check_desirable_crop(n, p, k, ph, rainfall):
    test_sample = np.array([[5.3, 76.0, 2720.2, 23.1]])
    test_sample_scaled = rain_scaler.transform(test_sample)

    test_values = np.array([[n, p, k, ph, rainfall]])
    test_values_scaled = scaler.transform(test_values)
    crop_prediction = lr_model.predict(test_values_scaled)

    return str(crop_prediction)

st.title("Crop Desirability Checker")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown('<div class="header-text">Do you want to predict the rainfall?</div>', unsafe_allow_html=True)

# Predict button
if st.button("Predict Rainfall"):
        st.success("Rainfall prediction logic will be implemented here.")

# Input columns
st.sidebar.header("Input Parameters")
n = st.sidebar.text_input("N (Nitrogen)")
p = st.sidebar.text_input("P (Phosphorus)")
k = st.sidebar.text_input("K (Potassium)")
ph = st.sidebar.text_input("pH")
rainfall = st.sidebar.text_input("Rainfall (mm)")

# enter your output hre
if st.sidebar.button("Check Desirable Crop"):
    desirable_crop = check_desirable_crop(float(n), float(p), float(k), float(ph), float(rainfall))
    st.success(f"The desirable crop is: {desirable_crop}")

