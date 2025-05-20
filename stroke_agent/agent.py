from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.helper import download_hugging_face_embeddings
from dotenv import load_dotenv
import os
from src.prompt import *
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from stroke_agent.tools.agent_tools import ecg_analyzer, prevention_retriever_tool, stroke_retriever_tool,explain_risk_tools,interpret_risk_scores

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_AGENT_API_KEY = os.environ.get('OPENAI_AGENT_API_KEY')

embeddings = download_hugging_face_embeddings()
os.environ["OPENAI_AGENT_API_KEY"] = OPENAI_AGENT_API_KEY
llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini", openai_api_key=OPENAI_AGENT_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# Shared embeddings and prompt setup
embeddings = download_hugging_face_embeddings()

# Stroke RAG chain
stroke_docsearch = PineconeVectorStore.from_existing_index(
    index_name="strokeindex",
    embedding=embeddings
)
stroke_retriever = stroke_docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
stroke_qa_chain = create_stuff_documents_chain(llm, prompt)
stroke_rag_chain = create_retrieval_chain(stroke_retriever, stroke_qa_chain)

# Prevention RAG chain
prevention_docsearch = PineconeVectorStore.from_existing_index(
    index_name="preventionindex",
    embedding=embeddings
)

prevention_retriever = prevention_docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
prevention_qa_chain = create_stuff_documents_chain(llm, prompt)
prevention_rag_chain = create_retrieval_chain(prevention_retriever, prevention_qa_chain)

bot_tools = [ecg_analyzer, stroke_retriever_tool, prevention_retriever_tool, explain_risk_tools,interpret_risk_scores]

llm_with_tools = llm.bind_tools(bot_tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

system_prompt = "You are a helpful assistant specialized in medical data analysis and ECG interpretation."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # Add the system prompt here
        ("human", "{input}"),
    ]
)

stroke_prompt = ("system",

"⚠️ You are a formatting-only assistant for medical data output. Your job is to apply HTML for readability—NOT to answer questions or interpret content. Follow these strict rules:" 

"\n\n🔒 **SCOPE RESTRICTION**"
"\n- Do not respond to anything outside of stroke, ECG, or stroke prevention."
"\n- Do not answer any follow-up questions, even if included in the content. Only insert follow-up *questions*, never answers."

"\n\n🖋️ **CONTENT PRESERVATION**"
"\n- NEVER summarize, shorten, rephrase, or omit any part of the tool’s output."
"\n- Preserve all manual bullet points, numbering, diagnostic names, ICD codes, and prediction scores exactly as-is."

"\n\n🧩 **FORMATTING RULES**"
"\n1. Use only these HTML tags: <b>, <i>, <u>, <h5>, <br>, <ul>, <ol>, <li>, <table>, <thead>, <tbody>, <tr>, <th>, <td>, <div>."
"\n2. Wrap long paragraphs with: <div style='line-height: 1.8;'> ... </div>."
"\n3. Use <br> for line breaks only in text, NOT inside lists or tables."
"\n4. Use:"
"\n   - <ul>/<li> for bullet lists"
"\n   - <ol>/<li> for ordered lists"
"\n   - Manual numbering only if already present. Do NOT convert to <ol>."
"\n5. Use <table> for any structured multi-column content (like JSON, lists of scores, or CSV-like data)."
"\n   - Add <thead> for headers, and <tbody> for rows."
"\n   - Style cells with inline CSS: borders and padding."
"\n   - Bold all headers."

"\n\n🎨 **EMPHASIS & STYLING**"
"\n- Use:"
"\n   - <b> for emphasis"
"\n   - <i> for highlights"
"\n   - <u> for underlined labels"
"\n   - <h5> for section titles"
"\n- Use emojis to enhance engagement and readability."

"\n\n🧠 **FOLLOW-UP BEHAVIOUR**"
"\n- After `ecg_analyzer` runs, append this question *at the end* of the response WITHOUT answering it:"
"\n   👉 <i>Would you like a further explanation of the Top 5 Predicted Conditions?</i>"
"\n- ONLY if the user answers yes, THEN the explanation tool (e.g., `explain_risk_tools`) should run and return formatted results."

"\n- After `explain_risk_tools` runs, append this follow-up *without answering it yet*:"
"\n   👉 <i>Would you like a tailored assessment of your risk calculation scores? Let's start with your ABCD score.</i>"

"\n🛑 DO NOT answer either follow-up question in the same step as asking it."

)

def chatbot(state: State):
    system_prompt = stroke_prompt
    # Prepend the system prompt to the messages
    messages = [system_prompt] + state["messages"]
    print("State Messages with System Prompt:", messages)
    return {"messages": [llm_with_tools.invoke(messages)]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=bot_tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}




