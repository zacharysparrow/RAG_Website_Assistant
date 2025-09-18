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
from pydantic import BaseModel, Field
#from langchain_community.document_loaders.blob_loaders import Blob
#from langchain_community.document_loaders.parsers import PyPDFParser
import logging

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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY)

# set up chroma vector database
chroma = Chroma(
    collection_name="documents",
    collection_metadata={"name": "documents", "description": "store documents"},
    persist_directory="./data",
    embedding_function=embeddings,
)
retriever = chroma.as_retriever(search_kwargs={"k": 9})  # Retrieve top 3 relevant docs

# set up prompt template
TEMPLATE = """
Here is the context:

<context>
{context}
</context>

And here is the question that must be answered using that context:

<question>
{input}
</question>

Please read through the provided context carefully. Then, analyze the question and attempt to find a
direct answer to the question within the context.

If you are able to find a direct answer, provide it and elaborate on relevant points from the
context using bullet points "-".

If you cannot find a direct answer based on the provided context, outline the most relevant points
that give hints to the answer of the question.

If no answer or relevant points can be found, the question is not related to the context, simply
state the following sentence without any additional text:

I couldn't find an answer to your question.

Output your response in plain text without using the tags <answer> and </answer> and ensure you are not
quoting context text in your response since it must not be part of the answer.
"""
PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

# set up simple chat history
contextualize_q_system_prompt = ("""
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. 

Be aware that the lastest user question may not reference context in the chat history at all, and may be about an entirely new topic.

Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
""")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# create retreiver and llm chains
llm_chain = create_stuff_documents_chain(llm, PROMPT)
#retrieval_chain = create_retrieval_chain(retriever, llm_chain)
history_aware_retrieval_chain = create_retrieval_chain(history_aware_retriever, llm_chain)

store = {}

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        last_k = 5
        if len(self.messages) > last_k:
            self.messages = self.messages[-last_k:]

    def clear(self) -> None:
        self.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
#        store[session_id] = ChatMessageHistory()
        store[session_id] = InMemoryHistory()
    return store[session_id]

def clear_memory(session_id: str):
    del store[session_id]
    return "Successful"

conversational_retrieval_chain = RunnableWithMessageHistory(
    history_aware_retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def ask_question(query: str, session_id: str) -> str:
    response = conversational_retrieval_chain.invoke({"input": query}, config={"configurable":{"session_id": session_id}})
    return (response["answer"], response["context"])
