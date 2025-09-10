import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
retriever = chroma.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 relevant docs

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

If no answer or relevant points can be found, or the question is not related to the context, simply
state the following sentence without any additional text:

I couldn't find an answer to your question.

Output your response in plain text without using the tags <answer> and </answer> and ensure you are not
quoting context text in your response since it must not be part of the answer.
"""
PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

# create retreiver and llm chains
llm_chain = create_stuff_documents_chain(llm, PROMPT)
retrieval_chain = create_retrieval_chain(retriever, llm_chain)

# core functions
#def store_document(documents: list[Document]) -> str:
#    chroma.add_documents(documents=documents)
#    return "document stored successfully"
#
#parser = PyPDFParser()
#
#def parse_pdf(file_content: bytes) -> list[Document]:
#    blob = Blob(data=file_content)
#    return [doc for doc in parser.lazy_parse(blob)]

def retrieve_document(query: str) -> list[Document]:
    return retriever.invoke(input=query)

def ask_question(query: str) -> str:
    response = retrieval_chain.invoke({"input": query})
    return (response["answer"], response["context"])
