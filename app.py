import streamlit as st
import pandas as pd
import joblib

# ====== CSS STYLING ======
st.markdown("""
    <style>
    /* Remove default padding and center alignment */
    .block-container {
        padding: 0 !important;
        margin: 0 auto;
        max-width: 100% !important;
    }

    body {
        background-color: #eef4fb;
    }

    /* NAVBAR */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #062f5f;
        padding: 10px 40px;
        color: white;
    }

    .navbar h1 {
        margin: 0;
        font-size: 24px;
        font-weight: bold;
    }

    .navbar a {
        color: white;
        text-decoration: none;
        margin-left: 30px;
        font-size: 16px;
    }

    .navbar a:hover {
        text-decoration: underline;
    }

    .main {
        background-color: #ffffff;
        padding: 30px 40px;
    }

    .title {
        font-size: 40px;
        color: #88227c;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 22px;
        color: #444;
        margin-bottom: 30px;
    }

    .info-box {
        background-color: #dce7f9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 30px;
        font-size: 16px;
    }

    .stButton>button {
        background-color: #b93c9f;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: auto;
        margin: 5px 0;
    }

    .stFileUploader {
        margin-bottom: 20px;
    }

    .st-expander {
        background-color: #f2f6fd;
        border-radius: 10px;
        margin-top: 10px;
    }

    .st-expanderHeader {
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ====== NAVBAR ======
st.markdown("""
    <div class="navbar">
        <h1>Predict Health</h1>
        <div>
            <a href="/?page=home">Home</a>
            <a href="/?page=data">Data Entry</a>
            <a href="/?page=assessment">Stroke Self Assessment</a>
            <a href="/?page=chatbot">Chatbot</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ====== PAGE ROUTING ======
query_params = st.query_params
page = query_params.get("page", "home")

# ====== MAIN WRAPPER ======
st.markdown('<div class="main">', unsafe_allow_html=True)

# ====== HOME PAGE ======
if page == "home":
    st.markdown('<div class="title">Welcome to Predict Health</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A medical assistant platform for time-series prediction and health support.</div>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1588776814546-ec7ee5ab8e19", use_container_width=True)

# ====== DATA ENTRY PAGE ======
elif page == "data":
    st.markdown('<div class="title">Patient Data Entry</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter or upload time-series health records for model prediction.</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>Note:</strong> Each entry contains vitals recorded at different time points.
            You can upload a CSV or enter them manually below.
        </div>
    """, unsafe_allow_html=True)

    # Load model
    model = joblib.load("decision_tree_model.pkl")

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
        with st.expander(f"🗂️ Entry {i + 1}", expanded=False):
            col1, col2, col3 = st.columns(3)
            keys = list(field.keys())
            for j, key in enumerate(keys):
                unique_key = f"{key}_{i}"
                col = [col1, col2, col3][j % 3]
                field[key] = col.number_input(
                    key.replace("_", " ").capitalize(), value=field[key], key=unique_key
                )

    if st.button("🏃 Run ML Model"):
        if not st.session_state.fields:
            st.error("No entries to process.")
        else:
            input_df = pd.DataFrame(st.session_state.fields)
            prediction = model.predict(input_df.values)
            st.subheader("Model Prediction")
            st.write(f"Prediction: {prediction[0]}")

# ====== STROKE ASSESSMENT PAGE ======
elif page == "assessment":
    st.markdown('<div class="title">Stroke Self Assessment</div>', unsafe_allow_html=True)
    st.markdown("Coming soon: A guided self-evaluation tool to assess stroke symptoms and risk.")

# ====== CHATBOT PAGE ======
elif page == "chatbot":
    st.markdown('<div class="title">Health Assistant Chatbot</div>', unsafe_allow_html=True)
    st.markdown("Coming soon: Talk to our AI assistant for help with symptoms, data, and predictions.")

# ====== CLOSE MAIN WRAPPER ======
st.markdown('</div>', unsafe_allow_html=True)
