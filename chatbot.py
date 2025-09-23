import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
#from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers import (MergerRetriever, ContextualCompressionRetriever)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (LongContextReorder, EmbeddingsRedundantFilter)

from pydantic import BaseModel, Field
#from langchain_community.document_loaders.blob_loaders import Blob
#from langchain_community.document_loaders.parsers import PyPDFParser
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.simple import Tool
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
#from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from ast import literal_eval

# error loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set up API key and models
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
#Here is the context:
#
#<context>
#{context}
#</context>
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

#If you do not know how to answer a question, first ensure that you have exhausted the available tools. If the provided tools do not return relevant context, reply by stating:
#
#I couldn't find an answer to your question.

## set up simple chat history
#contextualize_q_system_prompt = ("""
#Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. 
#
#Be aware that the lastest user question may not reference context in the chat history at all, and may be about an entirely new topic.
#In that case, ignore the chat history and return the question as is.
#
#Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
#""")
#
#contextualize_q_prompt = ChatPromptTemplate.from_messages(
#    [
#        ("system", contextualize_q_system_prompt),
#        MessagesPlaceholder("chat_history"),
#        ("human", "{input}"),
#    ]
#)
#
#history_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
#history_aware_retriever = MergerRetriever(retrievers=[history_retriever, retriever])
#redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
#reordering = LongContextReorder()
#pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reordering])
#compression_retriever_reordered = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=history_aware_retriever)
#
## create retreiver and llm chains
#llm_chain = create_stuff_documents_chain(llm, PROMPT)
##retrieval_chain = create_retrieval_chain(retriever, llm_chain)
##history_aware_retrieval_chain = create_retrieval_chain(history_aware_retriever, llm_chain)
#history_aware_retrieval_chain = create_retrieval_chain(compression_retriever_reordered, llm_chain)
#
#store = {}
#
#class InMemoryHistory(BaseChatMessageHistory, BaseModel):
#    messages: list[BaseMessage] = Field(default_factory=list)
#
#    def add_messages(self, messages: list[BaseMessage]) -> None:
#        self.messages.extend(messages)
#        last_k = 5
#        if len(self.messages) > last_k:
#            self.messages = self.messages[-last_k:]
#
#    def clear(self) -> None:
#        self.messages = []
#
#def get_session_history(session_id: str) -> BaseChatMessageHistory:
#    if session_id not in store:
##        store[session_id] = ChatMessageHistory()
#        store[session_id] = InMemoryHistory()
#    return store[session_id]
#
memory = MemorySaver()

def clear_memory(session_id: str):
    memory.delete_thread(session_id)
    return "successful"

class RetrieverInput(BaseModel):
    """Input to the retriever."""
    query: str = Field(description="Query to look up in the retriever")

def get_documents(query, retriever):
    docs = retriever.invoke(query)
    
    # Combine the content of all documents
    combined_content = "\n\n".join(doc.page_content for doc in docs)
    
    # Combine the metadata of all documents into a single dictionary or list
    combined_metadata = {}
    for doc in docs:
        for key, value in doc.metadata.items():
            # Append metadata values to lists for each key
            if key in combined_metadata:
                combined_metadata[key].append(value)
            else:
                combined_metadata[key] = [value]

    # Create a combined document with combined content and metadata
    combined_doc = {
        "content": combined_content,
        "metadata": combined_metadata
    }

    return combined_doc

def create_retriever_tool(retriever, name: str, description: str) -> Tool:
    """Creates a Tool for a given retriever with custom name and description."""
    return Tool(
        name=name,
        func=lambda query: get_documents(query, retriever),
        description=description,
        args_schema=RetrieverInput
        # response_format="content_and_artifact"
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
#conversational_agent = create_react_agent(llm, tools, prompt=PROMPT)
#agent_executor = AgentExecutor(agent=conversational_agent, tools=tools)
#conversational_retrieval_graph = AgentExecutor(agent=agent, tools=tools)

#conversational_retrieval_chain = RunnableWithMessageHistory(
#    history_aware_retrieval_chain,
#    get_session_history,
#    input_messages_key="input",
#    history_messages_key="chat_history",
#    output_messages_key="answer",
#)

def ask_question(query: str, session_id: str) -> str:
    response = conversational_agent.invoke({"messages": [query]}, config={"configurable":{"thread_id": session_id}})
#    response = conversational_agent.invoke({"messages": [query]})

    sources = []
    for r in response["messages"]:
        if isinstance(r, ToolMessage):
            curr_metadata = literal_eval(r.content)["metadata"]
            sources.append([dict(zip(curr_metadata,t)) for t in zip(*curr_metadata.values())])
        if isinstance(r, HumanMessage):
            sources = [] #only want sources relevant to most recent human message

    return_sources = [x for xs in sources for x in xs]
    print(return_sources)
#    response = conversational_retrieval_chain.invoke({"input": query}, config={"configurable":{"session_id": session_id}})
    return (response["messages"][-1].content, return_sources, None)
