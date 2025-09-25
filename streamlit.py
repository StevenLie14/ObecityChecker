import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Obesity Prediction App", layout="wide")
st.title("Obesity Prediction System")

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Choose a page", ["New Input", "Test Case", "Prediction History"])
    st.info("Predict your obesity level based on health & lifestyle data!")

if "history" not in st.session_state:
    st.session_state.history = []

def call_prediction_api(data):
    try:
        response = requests.post(f"{os.getenv("API_URL")}/predict", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def get_label_color(label):
    colors = {
        "Insufficient_Weight": "blue",
        "Normal_Weight": "green",
        "Overweight_Level_I": "orange",
        "Overweight_Level_II": "darkorange",
        "Obesity_Type_I": "red",
        "Obesity_Type_II": "darkred",
        "Obesity_Type_III": "black"
    }
    return colors.get(label, "gray")

if page.startswith("New"):
    st.subheader("Enter Your Health and Lifestyle Information")

    with st.form("obesity_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            Age = st.number_input("Age (Years)", min_value=14, max_value=150, value=25)
            Height = st.number_input("Height (in meters)", min_value=1., max_value=3., value=1.75, step=0.01)
            Weight = st.number_input("Weight (in kg)", min_value=40.0, max_value=250.0, value=70.0, step=0.1)
            family_history = st.radio("Family History of Overweight", ["yes", "no"], horizontal=True)

        with col2:
            NCP = st.number_input("Main Meals per Day", value=3.0)
            FCVC = st.slider("Frequency of Vegetable Consumption (Scale 1–3)", min_value=1.0, max_value=3.0, value=2.0, step=0.01)
            CH2O = st.slider("Water Intake (Liters per day) (Scale 0-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.01)
            FAF = st.slider("Physical Activity (hrs/week) (Scale 0-3)", min_value=0.0, max_value=3.0, value=1.5, step=0.01)
            TUE = st.slider("Technology Use (hrs/day) (Scale 0-3)", min_value=0.0, max_value=3.0, value=1.0, step=0.01)

        with col3:
            FAVC = st.radio("Frequent High-Calorie Food Consumption?", ["yes", "no"], horizontal=True)
            SCC = st.radio("Calorie Consumption Monitoring", ["yes", "no"], horizontal=True)
            CAEC = st.selectbox("Eating Between Meals", ["no", "Sometimes", "Frequently", "Always"])
            CALC = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])

        submitted = st.form_submit_button("Predict")
        
        
    if submitted:
        input_data = {
            "Gender": Gender, "Age": Age, "Height": Height, "Weight": Weight,
            "family_history_with_overweight": family_history, "FAVC": FAVC, "FCVC": FCVC, "NCP": NCP,
            "CAEC": CAEC, "CH2O": CH2O, "SCC": SCC,
            "FAF": FAF, "TUE": TUE, "CALC": CALC
        }
        with st.spinner("Analyzing your data..."):
            result = call_prediction_api(input_data)
            if "error" in result:
                st.error(f"Error : {result['error']}")
            else:
                label = result.get("prediction", "Unknown")
                st.success(f"Prediction: `{label}`")
                st.markdown(f"<h3 style='color:{get_label_color(label)}'>Your Obesity Level: {label}</h3>", unsafe_allow_html=True)
                st.session_state.history.append((input_data, label))

elif page.startswith("Test"):
    st.subheader("Test the API with Sample Data")

    example_data = {
        "Gender": "Male", "Age": 22, "Height": 1.75, "Weight": 85.0,
        "family_history_with_overweight": "yes", "FAVC": "yes", "FCVC": 2.5, "NCP": 3.0,
        "CAEC": "Sometimes", "CH2O": 2.0, "SCC": "no",
        "FAF": 1.5, "TUE": 1.0, "CALC": "Sometimes"
    }

    st.code(example_data, language="json")
    if st.button("Run Test Case"):
        with st.spinner("Sending test case..."):
            result = call_prediction_api(example_data)
            if "error" in result:
                st.error(f"Error : {result['error']}")
            else:
                label = result.get("prediction", "Unknown")
                st.success(f"Prediction: `{label}`")
                st.markdown(f"<h3 style='color:{get_label_color(label)}'> Test Case Result: {label}</h3>", unsafe_allow_html=True)
                
    example_data_2 = {
        "Gender": "Female", "Age": 35, "Height": 1.60, "Weight": 70.0,
        "family_history_with_overweight": "no", "FAVC": "no", "FCVC": 3.0, "NCP": 4.0,
        "CAEC": "no", "CH2O": 2.5, "SCC": "yes",
        "FAF": 3.0, "TUE": 0.5, "CALC": "Frequently"
    }
    
    st.code(example_data_2, language="json")
    if st.button("Run Test Case 2"):
        with st.spinner("Sending second test case..."):
            result = call_prediction_api(example_data_2)
            if "error" in result:
                st.error(f"Error : {result['error']}")
            else:
                label = result.get("prediction", "Unknown")
                st.success(f"Prediction: `{label}`")
                st.markdown(f"<h3 style='color:{get_label_color(label)}'> Test Case 2 Result: {label}</h3>", unsafe_allow_html=True)


elif page.startswith("Pre"):
    st.subheader("Prediction History")

    if st.session_state.history:
        for i, (data, label) in enumerate(st.session_state.history[::-1], 1):
            with st.expander(f"Prediction #{i}"):
                st.json(data)
                st.markdown(f"<b>Prediction:</b> <span style='color:{get_label_color(label)}'>{label}</span>", unsafe_allow_html=True)
    else:
        st.info("No predictions yet. Try entering data from the 'New Input' page.")
