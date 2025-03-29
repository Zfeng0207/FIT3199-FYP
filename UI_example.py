import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained ML model (replace with your model file path)
model = joblib.load("C:\Monash\FIT3164\FIT3199-FYP\decision_tree_model.pkl")

# Initialize session state for authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Login Page
if not st.session_state.logged_in:
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "MDS04" and password == "MDS04":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Entry", "Stroke Self-Assessment"])
st.session_state.page = page

# Home Page
if st.session_state.page == "Home":
    st.title("Welcome to the Healthcare Prediction System")
    st.write("This project uses machine learning to predict patient outcomes based on medical data. You can manually enter patient data or upload a CSV file for batch processing. Additionally, a stroke self-assessment tool is available to help users assess their risk factors.")

# Data Entry Page
elif st.session_state.page == "Data Entry":
    st.title("Patient Data Entry & ML Model Prediction")
    
    # Function to add a new set of input fields
    def add_new_fields():
        st.session_state.fields.append({
            "temperature": 0.0,
            "heartrate": 0.0,
            "resprate": 0.0,
            "o2sat": 0.0,
            "sbp": 0.0,
            "dbp": 0.0,
            "rhythm": 0,
            "pain": 0,
            "gender": 0,
            "anchor_age": 0,
            "anchor_year": 0,
            "anchor_year_group": 0,
            "year": 0.0,
            "month": 0.0,
            "day": 0.0,
            "hour": 0.0,
            "minute": 0.0,
            "second": 0.0
        })

    # Function to delete a field by its index
    def delete_entry(index):
        del st.session_state.fields[index]

    # Initialize session state to track the list of fields
    if "fields" not in st.session_state:
        st.session_state.fields = []

    # CSV file uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.fields = df.to_dict(orient="records")
        st.session_state.uploaded_file = uploaded_file
        st.success("CSV uploaded and data loaded.")
    else:
        if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
            st.session_state.fields = []  # Clear data if file is removed
            st.session_state.uploaded_file = None
            st.warning("CSV removed. Data cleared.")

    # Add a plus button to add new fields
    if st.button("➕ Add Another Entry"):
        add_new_fields()

    # Loop through existing fields and display them
    for i, field in enumerate(st.session_state.fields):
        with st.expander(f"Entry {i + 1}"):
            if st.button("❌", key=f"delete_{i}", help="Delete this entry"):
                delete_entry(i)
                st.rerun()

            inputs = {}
            for key in field:
                inputs[key] = st.number_input(key.capitalize(), value=field[key], key=f"{key}_{i}")

            st.session_state.fields[i].update(inputs)

    # Button to run the ML model
    if st.button("Run ML Model"):
        if not st.session_state.fields:
            st.error("No entries to process.")
        else:
            # Convert inputs to DataFrame for the model
            input_df = pd.DataFrame(st.session_state.fields)

            # Run the model prediction with the DataFrame
            prediction = model.predict(input_df.values)

            # Display the prediction
            st.subheader("Model Prediction")
            st.write(f"Prediction: {prediction[0]}")

# Stroke Self-Assessment Page
elif st.session_state.page == "Stroke Self-Assessment":
    st.title("Stroke Risk Assessment")
    
    st.write("Answer the following questions to assess your risk for stroke.")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
    heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
    smoking_status = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"])
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
    glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=250.0, value=100.0)
    
    if st.button("Assess Risk"):
        risk_score = 0
        if age > 55:
            risk_score += 2
        if hypertension == "Yes":
            risk_score += 2
        if heart_disease == "Yes":
            risk_score += 2
        if smoking_status == "Current Smoker":
            risk_score += 2
        if bmi > 30:
            risk_score += 2
        if glucose > 140:
            risk_score += 2
        
        if risk_score >= 6:
            st.error("High Risk: Please consult a doctor.")
        elif risk_score >= 3:
            st.warning("Moderate Risk: Monitor your health and consult a doctor if needed.")
        else:
            st.success("Low Risk: Maintain a healthy lifestyle.")