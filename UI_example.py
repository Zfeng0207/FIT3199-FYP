import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained ML model (replace with your model file path)
model = joblib.load("decision_tree_model.pkl")

# Initialize session state to track the list of fields
if "fields" not in st.session_state:
    st.session_state.fields = []

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

# Display title
st.title("Patient Data Entry & ML Model Prediction")

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