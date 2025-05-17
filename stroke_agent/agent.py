from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.helper import download_hugging_face_embeddings
from dotenv import load_dotenv
import os
from src.prompt import *
from langchain.tools import tool
from pydantic import BaseModel
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
import numpy as np
import matplotlib.pyplot as plt
import io
from pydantic import BaseModel
from stroke_agent.tools.agent_tools import ecg_analyzer, stroke_retriever_tool, prevention_retriever_tool

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

tools = [stroke_retriever_tool, prevention_retriever_tool, ecg_analyzer]

tools = [ecg_analyzer]

llm_with_tools = llm.bind_tools(tools)

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
                 
"Do not answer anything outside of stroke, ecg, or prevention."

"Do not summarize, shorten, or exclude any part of the Tool's output. Your only job is to apply HTML tags to improve readability without changing content. Always preserve all original bullet points, manual numbering, diagnostic details, and prediction scores exactly as-is."

"For any data with two or more columns (list, JSON, or CSV-like), convert it into a clean HTML table using <table>, <thead>, <tbody>, <tr>, <th>, and <td> with inline borders and padding. Bold headers."

"Use <div style='line-height: 1.8;'>...</div> for long text to improve readability."

"Use <br> for line breaks in long text, but do not use <br> for lists or tables."

"Try to show all information in either bullet or table format."

"Only use these tags: <b>, <i>, <u>, <h5>, <br>. No other HTML tags are allowed."

"For lists: use <ul>/<li> only for bullets, <ol>/<li> only for ordered lists, and manual numbering with no list tags. Never mix list formats in one response."

"Use <h5> for section titles, <u> for underlined labels, <b> for emphasis, and <i> for highlights. <br> for line breaks."

"Use emojis in headings and subheadings to improve engagement."

"Do not remove or modify medical prediction content. Use lots of emojis for engagement."

"After ecg_analyzer runs, always ask: 'Would you like a further explanation of the Top 5 Predicted Conditions?' "

"After explanation of the Top 5 Predicted Conditions always ask 'Would you like a personalized prevention plan based on these conditions?'"
)



def chatbot(state: State):
    system_prompt = stroke_prompt
    # Prepend the system prompt to the messages
    messages = [system_prompt] + state["messages"]
    print("State Messages with System Prompt:", messages)
    return {"messages": [llm_with_tools.invoke(messages)]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[ecg_analyzer])
tool_node = ToolNode(tools=[prevention_retriever_tool, stroke_retriever_tool, ecg_analyzer])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}




