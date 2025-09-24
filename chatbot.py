import os
#import streamlit as st
from streamlit import secrets
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.simple import Tool
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from pydantic import BaseModel, Field
import logging
from ast import literal_eval

# error loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set up API key and models
load_dotenv(find_dotenv())
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_API_KEY = secrets["GEMINI_API_KEY"]
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set")
    raise ValueError("GEMINI_API_KEY is not set")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="QUESTION_ANSWERING", google_api_key=GEMINI_API_KEY)
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)

# set up chroma vector database
chroma = Chroma(
    collection_name="documents",
    collection_metadata={"name": "documents", "description": "store documents"},
    persist_directory="./data",
    embedding_function=embeddings,
)
sci_retriever = chroma.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold":0.0, "filter":{"tag":"science"}})  # Retrieve top k relevant docs
web_retriever = chroma.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold":0.0, "filter":{"tag":"web"}})  # Retrieve top k relevant docs
cv_retriever = chroma.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold":0.0, "filter":{"tag":"cv"}})  # Retrieve top k relevant docs

# set up prompt template
TEMPLATE = """
You are Zachary Sparrow's (Zach) AI assistant designed to answer questions from hiring managers and recruiters about Zach's professional history, experience, and skills.
You are currently talking with a hiring manager or recruiter. Your goal is to convince them to give Zach an interview or a job.

The provided tools can be used to search Zach's CV, frequently asked questions, personal website (which contains information on personal projects, hobbies, etc.), and Zach's published peer review papers.
If needed, use the provided tools to gather information related to responding to the given question, prompt, or query.

Please use concise but complete answers, using bullet points "-" to organize your response only if needed.

Output your response in plain text without using tags and ensure you are not quoting context text in your response.

Here is the question:

<messages>
{messages}
</messages>
"""
PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

memory = MemorySaver()

def clear_memory(session_id: str):
    memory.delete_thread(session_id)
    return "successful"

class RetrieverInput(BaseModel):
    """Input to the retriever."""
    query: str = Field(description="Query to look up in the retriever")

def get_documents(query, retriever):
    docs = retriever.invoke(query)
    combined_content = "\n\n".join(doc.page_content for doc in docs)
    
    combined_metadata = {}
    for doc in docs:
        for key, value in doc.metadata.items():
            if key in combined_metadata:
                combined_metadata[key].append(value)
            else:
                combined_metadata[key] = [value]

    combined_doc = {
        "content": combined_content,
        "metadata": combined_metadata
    }

    return combined_doc

# custom retriever tool to also fetch references
def create_retriever_tool(retriever, name: str, description: str) -> Tool:
    return Tool(
        name=name,
        func=lambda query: get_documents(query, retriever),
        description=description,
        args_schema=RetrieverInput
    )

sci_tool = create_retriever_tool(
    sci_retriever,
    "scientific_paper_retriever",
    "Searches and returns excerpts from Zach's peer-reviewed scientific papers. Topics include: PEPPr, CASE21, Chemistry, Physics, machine learning, algorithms."
)

web_tool = create_retriever_tool(
    web_retriever,
    "personal_website_retriever",
    "Searches and returns excerpts from Zach's personal website and portfolio, including information about personal data science and machine learning/AI projects."
)

cv_tool = create_retriever_tool(
    cv_retriever,
    "resume_retriever",
    "Searches and returns excerpts from Zach's CV and answers to frequently asked questions."
)

tools = [sci_tool, web_tool, cv_tool]

# hook for chat history
def pre_model_hook(state) -> dict[str, list[BaseMessage]]:
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=2048,
        token_counter=count_tokens_approximately,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}

conversational_agent = create_react_agent(llm, tools, prompt=PROMPT, checkpointer=memory, pre_model_hook=pre_model_hook)

def ask_question(query: str, session_id: str) -> str:
    response = conversational_agent.invoke({"messages": [query]}, config={"configurable":{"thread_id": session_id}})

    sources = []
    for r in response["messages"]:
        if isinstance(r, ToolMessage):
            curr_metadata = literal_eval(r.content)["metadata"]
            sources.append([dict(zip(curr_metadata,t)) for t in zip(*curr_metadata.values())])
        if isinstance(r, HumanMessage):
            sources = [] #only want sources relevant to most recent human message

    return_sources = [x for xs in sources for x in xs]
    return (response["messages"][-1].content, return_sources, None)
