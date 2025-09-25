import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import requests
from streamlit.runtime.scriptrunner import get_script_run_ctx
import logging

# error loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ask(query: str, session_id: str) -> str:
    with st.spinner("Asking the chatbot..."):
        response = requests.get(f"{API_URL}/ask?query={query}&session_id={session_id}") #ZMS
    if response.status_code == 200:
        data = response.json()
        return (data["answer"], data["sources"])
    else:
        return "I couldn't find an answer to your question."

def reset_history(session_id: str):
    with st.spinner("Setting up the chat..."):
        status = requests.get(f"{API_URL}/reset_history?session_id={session_id}")
    if status == "successful":
        return "Reset successfull"
    
# set up streamlit page
load_dotenv(find_dotenv())
API_URL = os.getenv("API_URL")
if not API_URL:
    logger.error("API_URL is not set")
    raise ValueError("API_URL is not set")

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Zach's Personal AI Assistant")

if "initialized" not in st.session_state or not st.session_state.initialized:
    ctx = get_script_run_ctx()
    st.session_state.session_id = ctx.session_id
    st.session_state.initialized = True
    with st.chat_message(name="ai", avatar="ai"):
        st.write("Hello! I'm Zach's personal AI assistant. I can answer questions about Zach and his research, projects, and experience.")

query = st.chat_input(placeholder='''Type your question here... e.g., "Why should I hire Zach?"''')

if query:
    with st.chat_message("user"):
        st.write(query)
    try:
        answer, sources = ask(query, st.session_state.session_id)
    except:
        answer = "I've run into an issue. Please try again later!"
        sources = None
        pass

    with st.chat_message("ai"):
        st.write(answer)

        sources = [item for item in sources if item["title"] != None]
        seen_sources = set()
        filtered_sources = []
        for d in sources:
            t = tuple(sorted(d.items()))
            if t not in seen_sources:
                seen_sources.add(t)
                filtered_sources.append(d)

        if filtered_sources != []:
            expander = st.expander("Relevant work:")
            for source in filtered_sources:
                expander.write("- :small[" + source["authors"] + ", *" + source["title"] + "*, " + source["subject"] + "\n https://doi.org/" + source["doi"] +"]")

if st.button("Reset Session", key="button"):
    status = reset_history(st.session_state.session_id)
    st.session_state.initialized = False
    query = None
    del st.session_state["button"]
    st.rerun()
