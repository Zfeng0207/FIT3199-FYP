from flask import Flask, request, render_template, redirect, url_for, session, flash
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
# app = Flask(__name__)
# app = Flask(__name__, template_folder='src/templates')
app = Flask(__name__, 
            template_folder='src/templates', 
            static_folder='src/static')

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
FLASKAPP_API_KEY = os.environ.get('FLASKAPP_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["FLASKAPP_API_KEY"] = FLASKAPP_API_KEY

openai.api_base = "https://openrouter.ai//v1"
app.secret_key = FLASKAPP_API_KEY
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
# index_name = "darrenchenhw"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    temperature=0.4,
    max_tokens=500,
    model="qwen/qwen2.5-vl-3b-instruct:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    request_timeout=60,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def login():
    return render_template('login.html')

# later move this to a config file or database
users = {
    "itadmin": {"password": "password", "role": "staff"},
    "staff2": {"password": "pass123", "role": "staff"},
    "customer1": {"password": "password", "role": "customer"},
    "customer2": {"password": "custpass", "role": "customer"}
}

@app.route("/login", methods=["POST"])
def handle_login():
    username = request.form.get("username")
    password = request.form.get("password")

    # Check if the username exists in the users dictionary
    user = users.get(username)

    if user and user["password"] == password:
        session["user"] = username
        session["role"] = user["role"]
        return redirect(url_for("chat_page"))
    else:
        flash("Invalid credentials. Please try again.")
        return redirect(url_for("login"))

@app.route("/chat")
def chat_page():
    # Render the chat page
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

@app.route('/logout')
def logout():
    session.clear()  # or: session.pop('user', None)
    return redirect(url_for('login'))  # or wherever your login page is

@app.route('/data-entry')
def data_entry_routing():
    return render_template('data_entry.html')

@app.route('/chatbot')
def chatbot_routing():
    return render_template('chat.html')

@app.route('/self_assessment')
def self_assesment_routing():
    return render_template('self_assesment.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            flash('CSV uploaded successfully!', 'success')
            return redirect(url_for('upload_file'))
        else:
            flash('Only CSV files are allowed.', 'danger')
            return redirect(url_for('upload_file'))
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)