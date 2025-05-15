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
from stroke_agent.tools.agent_tools import ecg_analyzer, retriever_tool

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
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

tools = [retriever_tool, ecg_analyzer]
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
"Do not summarize, shorten, or exclude any part of the Tool's output. "
"Your only task is to enhance readability by adding appropriate HTML tags for structure and emphasis. "
"Always preserve every original bullet point, manual numbering, diagnostic detail, and prediction score exactly as-is. "
"Only apply the following HTML tags: <b>, <i>, <u>, <h5>, <div style='line-height: 1.2;'>...</div>. Use no other HTML tags. "
"When presenting lists, follow these rules strictly: "
"- If the Tool uses manual numbering (e.g., 1., 2., 3.), do NOT use any list tags like <ul>, <ol>, or <li>. "
"- If bullet points are used, wrap them only in <ul> and <li>—never use <ol> or manual numbering at the same time. "
"- If ordered numbering is used (not manual), wrap in <ol> and <li>—never mix with <ul> or manual numbers. "
"Do not combine <ul>, <ol>, and <li> together—only use one list format per response, based on the original structure. "
"Always wrap the entire response inside: <div style='line-height: 1.2;'>...</div>. "
"Use <h5> for section headers, <u> for underlining labels, <b> for emphasis, and <i> for scores or soft highlights. "
"Never delete, merge, or reword medical predictions or score data. "
"Use plenty of emojis to make the output fun and engaging for users. "
"After the ecg_analyzer tool runs, always ask the user: 'Would you like a further explanation of the Top 5 Predicted Conditions?' "
"After displaying the Top 5 conditions, also ask: 'Would you like a personalized prevention plan based on these conditions?' "
)


def chatbot(state: State):
    system_prompt = stroke_prompt
    # Prepend the system prompt to the messages
    messages = [system_prompt] + state["messages"]
    print("State Messages with System Prompt:", messages)
    return {"messages": [llm_with_tools.invoke(messages)]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[retriever_tool, ecg_analyzer])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}




