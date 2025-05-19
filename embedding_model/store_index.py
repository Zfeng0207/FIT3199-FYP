from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from helper import load_pdf_file, text_split, download_hugging_face_embeddings

# Load environment variables
load_dotenv()

# Set API Key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Download embedding model (shared for both indexes)
embeddings = download_hugging_face_embeddings()

# ---- INDEX 1: strokeindex ----
stroke_data = load_pdf_file(data='embedding_model/stroke_pdf')
stroke_chunks = text_split(stroke_data)

pc.create_index(
    name="strokeindex",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

stroke_docsearch = PineconeVectorStore.from_documents(
    documents=stroke_chunks,
    index_name="strokeindex",
    embedding=embeddings,
)

# ---- INDEX 2: preventionindex ----
prevention_data = load_pdf_file(data='embedding_model/prevention_pdf')
prevention_chunks = text_split(prevention_data)

pc.create_index(
    name="preventionindex",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

prevention_docsearch = PineconeVectorStore.from_documents(
    documents=prevention_chunks,
    index_name="preventionindex",
    embedding=embeddings,
)
