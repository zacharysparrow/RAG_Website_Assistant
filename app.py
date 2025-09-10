import streamlit as st
import requests

def ask(query: str) -> str:
    with st.spinner("Asking the chatbot..."):
        response = requests.get(f"{API_URL}/ask?query={query}")
    if response.status_code == 200:
        data = response.json()
        return (data["answer"], data["sources"])
    else:
        return "I couldn't find an answer to your question."

# set up streamlit page
API_URL = "http://localhost:8000"  # Change if deploying elsewhere
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Zach's Personal AI Assistant")

## file upload and document storage
#uploaded_files = st.file_uploader(
#    "Upload your PDF documents", type="pdf", accept_multiple_files=True
#)
#if uploaded_files:
#    files = [
#        ("files", (file.name, file.getvalue(), "application/pdf"))
#        for file in uploaded_files
#    ]
#    try:
#        with st.spinner("Uploading files..."):
#            response = requests.post(f"{API_URL}/documents/", files=files)
#        if response.status_code == 200:
#            st.success("Files uploaded successfully")
#            uploaded_files = None
#        else:
#            st.error("Failed to upload files")
#    except Exception as e:
#        st.error(f"Error uploading files: {e}")

# chat interface
with st.chat_message(name="ai", avatar="ai"):
    st.write("Hello! I'm Zach's personal AI assistant. I can answer questions about Zach and his research, projects, and experience.")

query = st.chat_input(placeholder="Type your question here...")

if query:
    with st.chat_message("user"):
        st.write(query)
    answer, sources = ask(query)
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
            st.write("For more information, please see the following:")
            for source in filtered_sources:
                st.write("- " + source["authors"] + ", " + source["title"] + ", " + source["subject"] + "\n https://doi.org/" + source["doi"])


