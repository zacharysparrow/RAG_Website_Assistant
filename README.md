# Retrieval-Augmented Generation (RAG) Website Assistant Chatbot

RAG chatbot designed to answer questions about me and my work. 

for development:
fastapi dev api.py
streamlit run app.py

make data/ directory for the chromadb
make documents/ directory for the documents you want the RAG to read


PDF -> markdown done with https://www.datalab.to/playground

chatbot_rag_env

TODO:
optimize chunking
add sldf paper when published
add boxes paper when on arxiv
fill out chromadb with:
- all my papers
- my website (update automatically?)
- my CV
- answers to common interview questions
test citations again -- k > 3, delete duplicate citations?
possible issue with assuming ambiguious context. Need to test more later.
switch to better model after all features implemented
host on aws
embed in website

Example:
https://github.com/mahdjourOussama/python-learning/tree/master/chatbot-rag
