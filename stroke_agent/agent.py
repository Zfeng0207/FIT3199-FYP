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
"Your role is only to enhance readability by adding appropriate HTML tags for structure and emphasis. "
"Preserve all original bullet points, numbering, scores, and notes exactly as they appear. "
"Apply formatting such as <b>, <i>, <u>to improve clarity. Don't use any other html tags! "
"Try to use bullets and numbering to make the text more readable. "
"Never delete or alter any diagnostic information or prediction scores."
"Make sure to use a lot of emojis to make the text more engaging and fun."
"After the ecg_analyzer tool is invoked, ask the user if they want further explanation of Top 5 Predicted Conditions"
"After the top 5 conditions are displayed, ask the user if they want a personalized prevention plan."
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




