import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth

#user authentication
names=['Peter Parker','Rebecca Miller']
usernames=['pparker','rmiller']
file_path=Path(__file__).parent / 'hashed_pw.pkl'
with file_path.open('rb') as file:
    hashed_passwords=pickle.load(file)

authenticator=stauth.Authenticate(names,usernames,hashed_passwords,'sales_dashboard','abcdef',cookie_expiry_days=0)
name,authentication_status,username=authenticator.login('Login','main')

if authentication_status is False:
    st.error('username/password incorrect')
if authentication_status is None:
    st.warning('please enter')
if authentication_status:
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Entry", "Stroke Self-Assessment"])
    authenticator.logout('Logout','sidebar')
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
        st.title("🩺 Stroke Risk Assessment")

        st.write("Source: https://www.stroke.org/en/about-stroke/stroke-risk-factors/stroke-risk-assessment")
        st.write("For each risk factor, select the box that applies to you. Select only one per risk factor.")

        # List of risk factors
        risk_factors = [
            "Is your blood pressure greater than 120/80 mmHg?",
            "Have you been diagnosed with atrial fibrillation?",
            "Is your blood sugar greater than 100 mg/dL?",
            "Is your body mass index greater than 25 kg/m²?",
            "Is your diet high in saturated fat, trans fat, sweetened beverages, salt, or excess calories?",
            "Is your total blood cholesterol greater than 160 mg/dL?",
            "Have you been diagnosed with diabetes mellitus?",
            "Do you get less than 150 minutes of moderate to vigorous-intensity activity per week?",
            "Do you have a personal or family history of stroke, TIA, or heart attack?",
            "Do you use tobacco or vape?"
        ]

        # Initialize session state for user responses and submission status
        if "responses" not in st.session_state:
            st.session_state.responses = [None] * len(risk_factors)
        if "submitted" not in st.session_state:
            st.session_state.submitted = False

        # Form for assessment
        with st.form("stroke_assessment_form"):
            for i, factor in enumerate(risk_factors):
                st.session_state.responses[i] = st.radio(
                    factor,
                    options=["Yes or unknown", "No"],
                    index=None if st.session_state.responses[i] is None else (0 if st.session_state.responses[i] == "Yes or unknown" else 1),
                    key=f"q_{i}"
                )

            # Submit button (inside form)
            submitted = st.form_submit_button("Submit")

        # Ensure all questions are answered before submission
        if submitted:
            if None in st.session_state.responses:
                st.error("⚠️ Please answer all questions before submitting.")
            else:
                st.session_state.submitted = True
                risk_score = sum(1 for response in st.session_state.responses if response == "Yes or unknown")

                st.subheader("Total Score:")
                st.write(f"**{risk_score} points**")

                # Risk Interpretation
                if risk_score >= 6:
                    st.error("⚠️ **High Risk:** You should consult a healthcare professional about reducing your stroke risk.")
                elif risk_score >= 3:
                    st.warning("⚠️ **Moderate Risk:** Consider making lifestyle changes and consult a doctor if needed.")
                else:
                    st.success("✅ **Low Risk:** Maintain a healthy lifestyle.")

                st.write("_If you scored higher in the 'Higher Risk' column or are unsure about your risk, ask your "
                        "healthcare professional about how to reduce your risk._")

        # Show "Clear All" button **outside** the form and **only after submission**
        if st.session_state.submitted:
            if st.button("Clear All"):
                st.session_state.responses = [None] * len(risk_factors)
                st.session_state.submitted = False  # Reset submission state
                st.rerun()  # Fully resets selections and updates the UI