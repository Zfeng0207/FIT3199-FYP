from flask import Flask, request, render_template, redirect, url_for, session, flash, send_file, Response
from werkzeug.utils import secure_filename
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import *
from dotenv import load_dotenv
import os
import json
import openai
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.flask_client import OAuth

app = Flask(__name__, template_folder='src/templates', static_folder='src/static')

load_dotenv()

# API Keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
FLASKAPP_API_KEY = os.environ.get('FLASKAPP_API_KEY')
OPENAI_AGENT_API_KEY = os.environ.get('OPENAI_AGENT_API_KEY')

# Set environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["FLASKAPP_API_KEY"] = FLASKAPP_API_KEY
os.environ["OPENAI_AGENT_API_KEY"] = OPENAI_AGENT_API_KEY
openai.api_base = "https://openrouter.ai//v1"
app.secret_key = FLASKAPP_API_KEY

# Embedding & Retrieval Setup
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini", openai_api_key=OPENAI_AGENT_API_KEY)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Demo users
users = {
    "itadmin": {"password": "password", "role": "staff"},
    "staff2": {"password": "pass123", "role": "staff"},
    "customer1": {"password": "password", "role": "customer"},
    "customer2": {"password": "custpass", "role": "customer"}
}

@app.route("/chat")
def chat_page():
    user = session.get('user')
    return render_template("chat.html", user=user, pretty=json.dumps(user, indent=4))

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("Input:", msg)

    if "plot sales" in msg.lower():
        memmap_meta_path = "ecg_dataset/memmap_meta.npz"
        memmap_path = "ecg_dataset/memmap.npy"
        memmap_meta = np.load(memmap_meta_path, allow_pickle=True)
        memmap_data = np.memmap(memmap_path, dtype=np.float32, mode='r')
        starts = memmap_meta["start"]
        lengths = memmap_meta["length"]
        original_shape = tuple(memmap_meta["shape"][0])
        ecg_data = memmap_data.reshape(original_shape)

        def visualize_12lead_ecg(ecg_data, patient_index=0):
            start_idx = starts[patient_index]
            length = lengths[patient_index]
            patient_data = ecg_data[start_idx:start_idx+length, :]
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
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
            plt.suptitle(f'12-Lead ECG - Patient #{patient_index+1}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            return fig

        fig = visualize_12lead_ecg(ecg_data, 0)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return send_file(img, mimetype='image/png')

    response = rag_chain.invoke({"input": msg})
    return Response(response["answer"], mimetype='text/plain')

@app.route('/data-entry')
def stroke_prediction_routing():
    user = session.get('user')
    return render_template('stroke_prediction.html', user=user, pretty=json.dumps(user, indent=4))

@app.route('/chatbot')
def chatbot_routing():
    user = session.get('user')
    return render_template('chat.html', user=user, pretty=json.dumps(user, indent=4))

@app.route('/self_assessment')
def self_assesment_routing():
    user = session.get('user')
    return render_template('self_assesment.html', user=user, pretty=json.dumps(user, indent=4))

@app.route('/abcd2_tia')
def abcd2_tia_routing():
    user = session.get('user')
    return render_template('abcd2_tia.html', user=user, pretty=json.dumps(user, indent=4))

@app.route('/about_us')
def about_us_routing():
    user = session.get('user')
    return render_template('about_us.html', user=user, pretty=json.dumps(user, indent=4))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        firstname = request.form.get('firstname')
        lastname = request.form.get('lastname')
        email = request.form.get('email')
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('thanks.html', name=firstname)
        else:
            return "Invalid file", 400
    return render_template('upload.html')

# Auth0 Integration
app.secret_key = env.get("APP_SECRET_KEY")
oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)

@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(redirect_uri=url_for("callback", _external=True))

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("https://" + env.get("AUTH0_DOMAIN") + "/v2/logout?" + urlencode({
        "returnTo": url_for("home", _external=True),
        "client_id": env.get("AUTH0_CLIENT_ID"),
    }, quote_via=quote_plus))

@app.route("/")
def home():
    user = session.get('user')
    try:
        with open('data/Country_Stroke_Count_with_ISO3_Standardised.csv', 'r', encoding='utf-8') as f:
            stroke_csv = f.read()
    except Exception as e:
        print("❌ Exception reading CSV:", e)
        return "Error reading stroke dataset!", 500
    return render_template('welcome.html', stroke_csv=stroke_csv, user=user)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))
