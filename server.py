from flask import Flask, jsonify, request, render_template, redirect, send_from_directory, url_for, session, flash
import joblib
import numpy as np
import pandas as pd
import torch

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import openai
from werkzeug.utils import secure_filename

#auth0 imports
import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv

from flask import Flask, request, send_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from flask import Response
from stroke_agent.agent import graph
from langchain.schema.messages import ToolMessage
import uuid
import stroke_agent.tools.agent_tools as agent_tools
import sys

# 1) Make sure we can import your modules
basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, "testing_calling_model"))

# 2) Monkey-patch __main__ for torch.load unpickling
import testing_calling_model.rnn_attention_model as rnn_attention_model
import __main__
__main__.RNNAttentionModel = rnn_attention_model.RNNAttentionModel
__main__.ConvNormPool    = rnn_attention_model.ConvNormPool
__main__.Swish           = rnn_attention_model.Swish
__main__.RNN             = rnn_attention_model.RNN

# 3) Now import predict_stroke exactly once from the module that defines it
from testing_calling_model.calling_model import predict_stroke

# 4) Configure Flask and uploads…
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__,
            template_folder='src/templates',
            static_folder='src/static')

UPLOAD_FOLDER = os.path.join(basedir, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    thread_id = session["thread_id"]
    import re
    import pandas as pd
    if "visualize" in msg.lower():
        return agent_tools.generate_patient_ecg_plot_html(msg)
         
    # Default: standard text query handled by LangGraph
    response = graph.invoke(
        {"messages": [("user", msg)]},
        {"configurable": {"thread_id": thread_id}}
    )
    return Response(response["messages"][-1].content, mimetype="text/html")

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
            return render_template('thanks.html', name=firstname)  # or redirect
        else:
            return "Invalid file", 400

    return render_template('upload.html')

#auth0 code
app.secret_key = env.get("APP_SECRET_KEY")

oauth = OAuth(app)

oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)

@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)
    )

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("home", _external=True),
                "client_id": env.get("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )

@app.route("/")
def home():
    user = session.get('user')
    print("User:", user)

    try:
        with open('data/Country_Stroke_Count_with_ISO3_Standardised.csv', 'r', encoding='utf-8') as f:
            stroke_csv = f.read()
    except Exception as e:
        print("❌ Exception reading CSV:", e)
        return "Error reading stroke dataset!", 500

    return render_template('welcome.html', stroke_csv=stroke_csv, user=user)

@app.route('/predict_memmap', methods=['POST'])
def predict_memmap():
    npy = request.files.get('npy_file')
    if not npy:
        return jsonify(error="No .npy file uploaded"), 400

    filename = secure_filename(npy.filename)
    npy_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    npy.save(npy_path)

    try:
        preds_df = predict_stroke(memmap_data=npy_path)
    except Exception as e:
        return jsonify(error=str(e)), 500

    # 1) reset index → makes an 'index' column
    preds_df = preds_df.reset_index()

    # 2) serialize entire table to a list of dicts
    rows = preds_df.to_dict(orient='records')

    # 3) write CSV (without the index column)
    csv_name = filename.replace('.npy', '_predictions.csv')
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_name)
    preds_df.drop(columns=['index']).to_csv(csv_path, index=False)

    download_url = url_for('download_predictions', filename=csv_name)

    return jsonify(rows=rows, download_url=download_url)



@app.route('/download_predictions/<filename>')
def download_predictions(filename):
    # serves the CSV with an attachment header
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,
                               as_attachment=True)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))