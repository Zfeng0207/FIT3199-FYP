import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("decision_tree_model.pkl")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Entry", "Stroke Self-Assessment"])
st.session_state.page = page

# Home Page
if st.session_state.page == "Home":
    user = st.experimental_user  # Store user info
    if user and hasattr(user, "name"):
        st.header(f"Hello {user.name} 👋")
        if hasattr(user, "picture"):
            st.image(user.picture, width=100)
    st.title("Welcome to the Healthcare Prediction System")
    st.write("This project uses machine learning to predict patient outcomes based on medical data.")

# Data Entry Page
elif st.session_state.page == "Data Entry":
    st.title("Patient Data Entry & ML Model Prediction")

    if "fields" not in st.session_state:
        st.session_state.fields = []

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.fields = df.to_dict(orient="records")
        st.success("CSV uploaded and data loaded.")

    if st.button("➕ Add Another Entry"):
        st.session_state.fields.append({
            key: 0.0 for key in [
                "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
                "rhythm", "pain", "gender", "anchor_age", "anchor_year",
                "anchor_year_group", "year", "month", "day", "hour", "minute", "second"
            ]
        })

    for i, field in enumerate(st.session_state.fields):
        with st.expander(f"Entry {i + 1}"):
            for key in field:
                unique_key = f"{key}_{i}"  # Ensure unique key for each input
                st.session_state.fields[i][key] = st.number_input(
                    key.capitalize(), value=field[key], key=unique_key
                )

    if st.button("Run ML Model"):
        if not st.session_state.fields:
            st.error("No entries to process.")
        else:
            input_df = pd.DataFrame(st.session_state.fields)
            prediction = model.predict(input_df.values)
            st.subheader("Model Prediction")
            st.write(f"Prediction: {prediction[0]}")

# Stroke Self-Assessment Page
elif st.session_state.page == "Stroke Self-Assessment":
    st.title("🩺 Stroke Risk Assessment")
    risk_factors = [
        "Is your blood pressure greater than 120/80 mmHg?",
        "Have you been diagnosed with atrial fibrillation?",
        "Is your blood sugar greater than 100 mg/dL?",
        "Is your body mass index greater than 25 kg/m²?",
        "Is your diet high in unhealthy fats or excess calories?",
        "Is your total blood cholesterol greater than 160 mg/dL?",
        "Have you been diagnosed with diabetes mellitus?",
        "Do you get less than 150 minutes of exercise per week?",
        "Do you have a family history of stroke or heart attack?",
        "Do you use tobacco or vape?"
    ]

    if "responses" not in st.session_state:
        st.session_state.responses = [None] * len(risk_factors)

    with st.form("stroke_assessment_form"):
        for i, factor in enumerate(risk_factors):
            st.session_state.responses[i] = st.radio(factor, ["Yes or unknown", "No"], key=f"q_{i}")
        submitted = st.form_submit_button("Submit")

    if submitted:
        risk_score = sum(1 for response in st.session_state.responses if response == "Yes or unknown")
        st.subheader("Total Score:")
        st.write(f"**{risk_score} points**")
        if risk_score >= 6:
            st.error("⚠️ High Risk: Consult a healthcare professional.")
        elif risk_score >= 3:
            st.warning("⚠️ Moderate Risk: Consider lifestyle changes.")
        else:
            st.success("✅ Low Risk: Maintain a healthy lifestyle.")