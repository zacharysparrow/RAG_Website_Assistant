from fastapi import FastAPI
from chatbot import ask_question, clear_memory
from pydantic import BaseModel
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create fastapi instance
app = FastAPI(
    title="RAG Chatbot",
    description="A Retrieval-Augmented Generation (RAG) chatbot using Google Gemini. Ask it questions about me, my experience, or my science.",
    version="0.1",
)

# pydantic models
class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list
    error: str = None

# api endpoints
@app.get("/")
def read_root():
    return {
        "service": "RAG Chatbot using Google Gemini",
        "description": "Welcome to RAG Chatbot API",
        "status": "running",
    }

@app.get("/reset_history")
def reset_history(session_id: str):
    status = clear_memory(session_id)
    return status

@app.get("/ask")
def ask(query: str, session_id: str) -> AskResponse:
    try:
        answer, context, history = ask_question(query, session_id)
        sources = []
        for source in context:
            try:
                doi = source["doi"]
            except:
                doi = None
            try:
                authors = source["author"]
            except:
                authors = None
            try:
                title = source["title"]
            except:
                title = None
            try:
                subject = source["subject"]
            except:
                subject = None
            sources.append({"title": title, "doi": doi, "authors": authors, "subject": subject})

        return {"query": query, "answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Error asking question: {e}", exc_info=True)
        return {"error": str(e), "query": query, "answer": "", "sources": ""}

