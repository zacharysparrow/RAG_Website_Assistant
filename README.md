# Retrieval-Augmented Generation (RAG) Website AI Agent
clean up .py files
make sure requirements.txt is complete

RAG chatbot made to answer questions about me and my work. 

for development:
fastapi dev api.py
streamlit run app.py

make data/ directory for the chromadb
make documents/ directory for the documents you want the RAG to read

PDF -> markdown done with https://www.datalab.to/playground

chatbot_rag_env

Hosting fastAPI and small chromadb on render.com
 - one benefit of render is that we can use memory for chat history, can't for serverless hosting
Hosting streamlit frontend on community cloud
Automatically updated with push to github

TODO:

documents:
add sldf paper when published
add boxes paper when on arxiv/published
fill out zach_info.md with answers to common interview questions
info on how the chatbot was made

