import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_community.document_loaders.blob_loaders import Blob
#from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders.pdf import PyPDFLoader
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

# set up chroma vector database
chroma = Chroma(
    collection_name="documents",
    collection_metadata={"name": "documents", "description": "store documents"},
    persist_directory="./data",
    embedding_function=embeddings,
)

loaded_docs = chroma.get()
loaded_doc_names = [doc["source"] for doc in loaded_docs["metadatas"]]

def store_document(documents: list[Document]) -> str:
    chroma.add_documents(documents=documents)
    return "document stored successfully"

documents = []
docs_path = (
    "./documents"
)

pdf_files = []
for filename in os.listdir(docs_path):
    if filename.endswith('.pdf'):
        pdf_files.append(os.path.join(docs_path, filename))

# checking for duplicate file names and adding
for doc in pdf_files:
    loader = PyPDFLoader(doc)
    content = loader.load()
    if doc not in loaded_doc_names: 
        print(f"{doc} added to database.")
        status = store_document(content)

