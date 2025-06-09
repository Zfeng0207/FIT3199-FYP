import os
import io
import base64
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from langchain.tools import tool

# from stroke_agent.stroke_data.icd_mapping_dict import icd_code_dict
import stroke_agent.agent as agent

icd_code_dict = {
    "I481": "Persistent atrial fibrillation",
    "I4891": "Atrial fibrillation, unspecified",
    "I110": "Hypertensive heart disease with heart failure",
    "I120": "Hypertensive chronic kidney disease with stage 5 chronic kidney disease or end stage renal disease",
    "I132": "Hypertensive heart and chronic kidney disease with heart failure and with stage 5 CKD or ESRD",
    "I210": "ST elevation (STEMI) myocardial infarction of anterior wall",
    "I200": "Unstable angina",
    "I255": "Ischemic cardiomyopathy",
    "I7025": "Atherosclerosis of native arteries of other extremities with ulceration",
    "I447": "Left bundle-branch block, unspecified",
    "I451": "Other and unspecified right bundle-branch block",
    "I440": "First degree atrioventricular block",
    "R000": "Tachycardia, unspecified",
    "R001": "Bradycardia, unspecified",
    "I5043": "Acute on chronic combined systolic and diastolic heart failure",
    "I081": "Rheumatic disorders of both mitral and tricuspid valves",
    "I340": "Nonrheumatic mitral (valve) insufficiency",
    "I359": "Nonrheumatic aortic valve disorder, unspecified",
    "I078": "Other rheumatic tricuspid valve diseases",
    "I428": "Other cardiomyopathies",
    "E1129": "Type 2 diabetes mellitus with other diabetic kidney complication",
    "E103": "Type 1 diabetes mellitus with ophthalmic complications",
    "E660": "Obesity due to excess calories",
    "N186": "End stage renal disease",
    "D631": "Anemia in chronic kidney disease",
    "D65": "Disseminated intravascular coagulation [defibrination syndrome]",
    "C925": "Acute myelomonocytic leukemia",
    "A40": "Streptococcal sepsis",
    "A419": "Sepsis, unspecified organism",
    "R570": "Cardiogenic shock"
}

from pydantic import BaseModel
from langchain.tools import tool

# Schema for the input question
class RagToolSchema(BaseModel):
    question: str

# Stroke RAG Tool
@tool(args_schema=RagToolSchema, return_direct=True)
def stroke_retriever_tool(question: str) -> str:
    """
    A tool for retrieving information on Guidelines for Management of Stroke
    """
    print("INSIDE STROKE RETRIEVER NODE")
    response = agent.stroke_rag_chain.invoke({"input": question})
    return response.get("answer")

# Prevention RAG Tool
@tool(args_schema=RagToolSchema)
def prevention_retriever_tool(question: str) -> str:
    """
    A tool for retrieving information on **technologies and methods for stroke monitoring and early detection**,
    particularly focusing on the **pre-hospital or continuous monitoring** context. Use this tool to learn about the
    importance of early detection within the 'Golden Hour', the **limitations of current stroke identification**
    methods, the need for continuous monitoring (e.g., during sleep, sedation, or for those living alone), and various
    **specific technological approaches** being explored or used for stroke detection (e.g., EEG, NIRS, Doppler ultrasound,
    motion sensors, pulse monitoring). It covers the present and future landscape of technology aimed at
    improving stroke detection time and patient outcomes.
    """
    print("INSIDE PREVENTION RETRIEVER NODE")
    response = agent.prevention_rag_chain.invoke({"input": question})
    return response.get("answer")

# Get the absolute path to the current file (agent_tools.py)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to reach stroke_agent/
stroke_agent_dir = os.path.dirname(current_file_dir)
# Build the path to the CSV
vital_data_path = os.path.join(stroke_agent_dir, "stroke_data", "vitals_data.csv")

# Load and preprocess data
print("Loading and preprocessing vitals data...")
vitals_data = pd.read_csv(vital_data_path)

# Convert the stringified list columns to actual lists
vitals_data["prediction_score"] = vitals_data["prediction_score"].apply(ast.literal_eval)
vitals_data["res"] = vitals_data["res"].apply(ast.literal_eval)

# Pre-compute sorted scores for each row
print("Pre-computing sorted scores...")
vitals_data['sorted_scores'] = vitals_data['prediction_score'].apply(
    lambda scores: sorted(zip(icd_code_dict.keys(), scores), key=lambda x: x[1], reverse=True)
)

# Set index for faster lookups
vitals_data.set_index(['subject_id', 'admission_id'], inplace=True)

# Cache for frequently accessed data
_icd_codes = list(icd_code_dict.keys())
_table_header = (
    "<table border='1' cellspacing='0' cellpadding='6' style='border-collapse: collapse;'>"
    "<tr><th>Rank</th><th>ICD Code</th><th>Condition</th><th>Prediction Score</th></tr>"
)
_table_footer = "</table>"

# Pre-compute static HTML parts
_notes_section = (
    "**📝 Notes:**\n"
    "- These results are generated by a deep learning model trained for multilabel ECG classification.\n"
    "- Additional diagnoses were predicted with lower confidence and are excluded from this summary."
)

_response_template = (
    "📈 <b>ECG Analysis Summary</b><br>"
    "<b>Subject ID:</b> `{subject_id}`<br>"
    "<b>Admission ID:</b> `{admission_id}`<br><br>"
    "🔍 <b>Top 5 Predicted Conditions:</b><br>{top5_table}<br><br>"
    "{notes_section}<br><br>"
    "Would you like to know how you can monitor your health for stroke assessment?"
)

class AnalyzerToolSchema(BaseModel):
    subject_id: int = Field(..., description="The subject ID from the hospital records")
    admission_id: int = Field(..., description="The admission ID corresponding to the hospital stay")

@tool(args_schema=AnalyzerToolSchema)
def ecg_analyzer(subject_id: int, admission_id: int):
    """
    Generates a detailed ECG prediction summary for a given subject and admission ID, based on deep learning model outputs.

    The tool returns:
    - Subject and admission identifiers.
    - The Top 5 predicted ICD conditions with confidence scores in a table format.
    - Notes on model usage and prediction confidence.

    🔒 Do not summarize, interpret, or omit any prediction content.
    """
    try:
        # Get pre-computed sorted scores
        score_pairs = vitals_data.loc[(subject_id, admission_id), 'sorted_scores']
    except KeyError:
        return f"No data found for subject ID {subject_id} and admission ID {admission_id}."

    # Generate table rows for top 5
    table_rows = []
    for i, (code, score) in enumerate(score_pairs[:5]):
        table_rows.append(
            f"<tr><td>{i + 1}</td><td>{code}</td><td>{icd_code_dict[code]}</td><td>{score:.2f}</td></tr>"
        )

    # Create the table
    top5_table = _table_header + "".join(table_rows) + _table_footer

    # Use pre-computed template
    return _response_template.format(
        subject_id=subject_id,
        admission_id=admission_id,
        top5_table=top5_table,
        notes_section=_notes_section
    )

import re
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Response
from markupsafe import Markup

def generate_patient_ecg_plot_html(msg: str) -> Response:
    # Extract subject_id and admission_id using regex
    subject_match = re.search(r"subject.*?(1\d{7})", msg, re.IGNORECASE)
    admission_match = re.search(r"admission.*?(5\d{7})", msg, re.IGNORECASE)

    if subject_match and admission_match:
        subject_id = int(subject_match.group(1))
        admission_id = int(admission_match.group(1))
    else:
        return Response("❌ Please provide a valid subject ID and admission ID.", mimetype="text/html")

    # Load vital data
    vital_data = pd.read_csv(vital_data_path)

    # Find patient row
    patient_row = vital_data[(vital_data['subject_id'] == subject_id) & 
                             (vital_data['admission_id'] == admission_id)]

    if patient_row.empty:
        return Response("❌ Patient not found in vitals_data.csv.", mimetype="text/html")

    # Extract index and ECG data slice
    patient_index = patient_row.index[0]
    start_idx = patient_row['start'].values[0]
    length = patient_row['length'].values[0]

    ecg_data_path = os.path.join(stroke_agent_dir, "stroke_data", "ecg_data.npy") # Build relative paths to both files
    ecg_data = np.load(ecg_data_path).reshape((100000, 12))
    patient_data = ecg_data[start_idx:start_idx + length, :]

    # ECG leads
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Create 3x4 plot
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < 12:
            ax.plot(patient_data[:, i])
            ax.set_title(f'Lead {lead_names[i]}')
            ax.grid(True, alpha=0.3)

            y_range = np.max(patient_data[:, i]) - np.min(patient_data[:, i])
            scale_bar = y_range * 0.2
            ax.plot([10, 10], [np.min(patient_data[:, i]), np.min(patient_data[:, i]) + scale_bar], 'k-', linewidth=2)

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.suptitle(f'12-Lead ECG - Subject {subject_id}, Admission {admission_id}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Convert plot to base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plt.close()
    # Encode image to base64
    img_base64 = base64.b64encode(img.read()).decode('utf-8')

    # Generate HTML with embedded image
    html_content = f"""
    <html>
            <img src="data:image/png;base64,{img_base64}" alt="ECG Plot" style="max-width:100%; height:auto;">
    </html>
    """

    return html_content

from pydantic import BaseModel, Field
from typing import Optional

from pydantic import BaseModel, Field
from langchain.tools import tool


@tool
def explain_risk_tools() -> str:
    """
    Explains the ASCVD Risk Estimator and ABCD² Score and provides a comparison table.
    """

    print("INSIDE RISK CALCULATORS EXPLANATION NODE")

    return """
✅ Based on your condition, here's how we'll assess your stroke and cardiovascular risk:

📊 <b>ASCVD Risk Estimator (Atherosclerotic Cardiovascular Disease)</b>:  
Estimates your 10-year and 30-year risk of heart attack or stroke based on:
<ul>
  <li>Age, sex, race</li>
  <li>Blood pressure</li>
  <li>Cholesterol levels</li>
  <li>Diabetes, smoking, medications</li>
</ul>
The output guides how aggressively we should manage your lifestyle and medical therapy.

🧠 <b>ABCD² Score for TIA (Transient Ischemic Attack)</b>:  
Predicts short-term stroke risk after a TIA using:
<ul>
  <li>Age ≥60 (1 point)</li>
  <li>BP ≥140/90 mmHg (1 point)</li>
  <li>Clinical symptoms: weakness (2), speech disturbance (1)</li>
  <li>Duration ≥60 min (2), 10–59 min (1)</li>
  <li>Diabetes (1 point)</li>
</ul>
A high ABCD² score indicates a higher short-term stroke risk.

<br><br>
<h4>📊 Comparison Table: ASCVD vs ABCD² Score</h4>
<table border="1" cellpadding="6" style="border-collapse: collapse; text-align: left;">
  <tr>
    <th>Feature</th>
    <th>ASCVD Risk Estimator</th>
    <th>ABCD² Score</th>
  </tr>
  <tr>
    <td><b>Purpose</b></td>
    <td>Estimate long-term risk of heart attack or stroke</td>
    <td>Predict short-term stroke risk after a TIA</td>
  </tr>
  <tr>
    <td><b>Time Horizon</b></td>
    <td>10 and 30 years</td>
    <td>2 to 7 days</td>
  </tr>
  <tr>
    <td><b>Inputs</b></td>
    <td>Age, sex, race, BP, cholesterol, diabetes, smoking</td>
    <td>Age, BP, symptoms, duration, diabetes</td>
  </tr>
  <tr>
    <td><b>Scoring Output</b></td>
    <td>% probability of cardiovascular event</td>
    <td>0–7 point score</td>
  </tr>
  <tr>
    <td><b>Use Case</b></td>
    <td>Primary prevention in general population</td>
    <td>Emergency risk stratification after TIA</td>
  </tr>
</table>

<br>
👉 <i>Would you like a tailored assessment of your risk calculation scores? Let's start with your ABCD score.</i>
"""

from pydantic import BaseModel
from langchain.tools import tool  # or your custom decorator
import re

from pydantic import BaseModel, Field

class RiskInterpretationSchema(BaseModel):
    ten_year_total_cvd: float = Field(..., ge=0, description="10-year total cardiovascular disease risk (%)")
    ten_year_ascvd: float = Field(..., ge=0, description="10-year ASCVD risk (%)")
    ten_year_heart_failure: float = Field(..., ge=0, description="10-year heart failure risk (%)")
    ten_year_chd: float = Field(..., ge=0, description="10-year coronary heart disease risk (%)")
    ten_year_stroke: float = Field(..., ge=0, description="10-year stroke risk (%)")

    thirty_year_total_cvd: float = Field(..., ge=0, description="30-year total cardiovascular disease risk (%)")
    thirty_year_ascvd: float = Field(..., ge=0, description="30-year ASCVD risk (%)")
    thirty_year_heart_failure: float = Field(..., ge=0, description="30-year heart failure risk (%)")
    thirty_year_chd: float = Field(..., ge=0, description="30-year coronary heart disease risk (%)")
    thirty_year_stroke: float = Field(..., ge=0, description="30-year stroke risk (%)")

@tool(args_schema=RiskInterpretationSchema)
def interpret_risk_scores(
    ten_year_total_cvd: float,
    ten_year_ascvd: float,
    ten_year_heart_failure: float,
    ten_year_chd: float,
    ten_year_stroke: float,
    thirty_year_total_cvd: float,
    thirty_year_ascvd: float,
    thirty_year_heart_failure: float,
    thirty_year_chd: float,
    thirty_year_stroke: float
) -> str:
    """
    Interprets cardiovascular and stroke risk results from structured input scores.
    """

    stroke_risk_score = ten_year_stroke

    return f"""
<h5>🧠 Stroke Risk Interpretation</h5>

<b>Known Stroke-Related Diagnoses (ICD Codes):</b>
<ul>
  <li><b>N186</b> – End stage renal disease – <i>Score: 0.99</i></li>
  <li><b>I132</b> – Hypertensive heart and chronic kidney disease with heart failure and with stage 5 CKD or ESRD – <i>Score: 0.97</i></li>
  <li><b>E103</b> – Type 1 diabetes mellitus with ophthalmic complications – <i>Score: 0.95</i></li>
  <li><b>I081</b> – Rheumatic disorders of both mitral and tricuspid valves – <i>Score: 0.89</i></li>
  <li><b>I447</b> – Left bundle-branch block, unspecified – <i>Score: 0.80</i></li>
</ul>


<b>Combined with model-based risk scores, this suggests the patient has both a clinical history and elevated probability of future stroke events.</b><br><br>

<h5>📊 10-Year Predicted Risk:</h5>
<ul>
  <li>Stroke: <b>{ten_year_stroke:.2f}%</b></li>
  <li>Total CVD: {ten_year_total_cvd:.2f}%</li>
  <li>ASCVD: {ten_year_ascvd:.2f}%</li>
</ul>

<h5>⏳ 30-Year Predicted Stroke Risk:</h5>
<ul>
  <li><b>{thirty_year_stroke:.2f}%</b> (significantly elevated)</li>
</ul>

🧠 <i>This combination of clinical codes and risk indicators places the patient in a high vigilance category for stroke monitoring.</i><br><br>

<b>✅ Would you like to receive a prevention plan tailored to your risk profile?</b>

"""
