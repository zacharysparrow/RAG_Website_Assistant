import os
import re
import json
import time
import logging

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
#from langchain_community.document_loaders.blob_loaders import Blob
#from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

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
print(loaded_doc_names)

def store_document(documents: list[Document]) -> str:
    chroma.add_documents(documents=documents)
    return "document stored successfully"

documents = []
docs_path = (
    "./documents/markdown"
)

md_files = []
for filename in os.listdir(docs_path):
    if filename.endswith('.md'):
        md_files.append(os.path.join(docs_path, filename))

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5")
]

with open('documents/metadata.json', 'r') as f:
    json_file = json.load(f)

for doc in md_files:
    loader = UnstructuredMarkdownLoader(doc, mode="single")
    content = loader.load()[0] #content isn't being loaded with headers
    json_string = doc.split("/")[-1].replace(".md","")
    doc_metadata = json_file[json_string]
    doc_metadata["source"] = json_string

    if doc_metadata["source"] not in loaded_doc_names: 
        tic = time.time()
        print(f"Adding {doc}...")

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        texts = markdown_splitter.split_text(content.page_content)
        chunk_size = 4000
        chunk_overlap = chunk_size*0.2
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False, separators=['. ']
        )
        all_text = []
        
        for t in texts:
            all_text.append(text_splitter.split_text(t.page_content))
        all_texts = [re.sub('(.)\n(?! \n)', r'\1 ', item[2:]+".") for sublist in all_text for item in sublist]
        
#        doc_list = []
        for t in all_texts:
#            doc_list.append(Document(page_content=t, metadata=doc_metadata))
            status = store_document([Document(page_content=t, metadata=doc_metadata)])
            time.sleep(5)

#        status = store_document(doc_list)
        
        toc = time.time()
        print(toc - tic)
        print(f"{doc} added to database.")

