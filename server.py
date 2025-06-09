from flask import Flask, jsonify, request, render_template, redirect, send_from_directory, url_for, session

from dotenv import load_dotenv
from app.prompt import *
import os
import openai
from werkzeug.utils import secure_filename

#auth0 imports
import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

from flask import Flask, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from flask import Response
from stroke_agent.agent import graph

import uuid
import stroke_agent.tools.agent_tools as agent_tools
import sys
import time
# 1) Make sure we can import your modules
basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, "model_inferencing"))

# 2) Monkey-patch __main__ for torch.load unpickling
import model_inferencing.rnn_attention_model as rnn_attention_model
import __main__
__main__.RNNAttentionModel = rnn_attention_model.RNNAttentionModel
__main__.ConvNormPool    = rnn_attention_model.ConvNormPool
__main__.Swish           = rnn_attention_model.Swish
__main__.RNN             = rnn_attention_model.RNN

# 3) Now import predict_stroke exactly once from the module that defines it
from model_inferencing.calling_model import predict_stroke

# 4) Configure Flask and uploads…
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__,
            template_folder='app/templates',
            static_folder='app/static')

UPLOAD_FOLDER = os.path.join(basedir, "app/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


load_dotenv()

# API Keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
FLASKAPP_API_KEY = os.environ.get('FLASKAPP_API_KEY')
OPENAI_AGENT_API_KEY = os.environ.get('OPENAI_AGENT_API_KEY')
LANGSMITH_TRACING= os.environ.get('LANGSMITH_TRACING')
LANGSMITH_ENDPOINT=os.environ.get('LANGSMITH_ENDPOINT')
LANGSMITH_API_KEY=os.environ.get('LANGSMITH_API_KEY')
LANGSMITH_PROJECT=os.environ.get('LANGSMITH_PROJECT')

# Set environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
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
    if any(word in msg.lower() for word in ["visualize", "plot", "graph", "chart", "diagram", "draw", "display", "render", "illustrate", "show"]):
        print("ECG VISUALIZER INVOKED")
        time.sleep(6)
        return agent_tools.generate_patient_ecg_plot_html(msg)
         
    # Default: standard text query handled by LangGraph
    response = graph.invoke(
        {"messages": [("user", msg)]},
        {"configurable": {"thread_id": "fwew2f12"}}
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
        with open('app/data/Country_Stroke_Count_with_ISO3_Standardised.csv', 'r', encoding='utf-8') as f:
            stroke_csv = f.read()
    except Exception as e:
        print("❌ Exception reading CSV:", e)
        return "Error reading stroke dataset!", 500

    return render_template('welcome.html', stroke_csv=stroke_csv, user=user)

from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    csv_file = request.files.get('csv_file')
    if not csv_file:
        return jsonify(error="No CSV file uploaded"), 400

    # Save the uploaded file
    filename = secure_filename(csv_file.filename)
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    csv_file.save(csv_path)

    try:
        # Run prediction using the simplified one-CSV method
        preds_df = predict_stroke(csv_with_ecg=csv_path, full_model='testing_calling_model/full_model.pkl')
    except Exception as e:
        return jsonify(error=str(e)), 500

    # Prepare output
    preds_df = preds_df.reset_index()
    rows = preds_df.to_dict(orient='records')

    # Save predictions to CSV (drop index column)
    output_csv_name = filename.replace('.csv', '_predictions.csv')
    output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], output_csv_name)
    preds_df.drop(columns=['index']).to_csv(output_csv_path, index=False)

    download_url = url_for('download_predictions', filename=output_csv_name)

    return jsonify(rows=rows, download_url=download_url)



@app.route('/download_predictions/<filename>')
def download_predictions(filename):
    # serves the CSV with an attachment header
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,
                               as_attachment=True)

@app.route('/download_sample')
def download_sample():
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        'df_split_1.csv',
        as_attachment=True,
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))