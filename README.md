# Retrieval-Augmented Generation (RAG) Website Assistant Chatbot

RAG chatbot designed to answer questions about me and my work. 

for development:
fastapi dev api.py
streamlit run app.py

make data/ directory for the chromadb
make documents/ directory for the documents you want the RAG to read



chatbot_rag_env

TODO:
fill out chromadb with:
- all my papers
- my website (update automatically?)
- my CV
- answers to common interview questions
want to only cite papers if they are actually relevant...
test citation robustness
switch to better model after all features implemented
test it all
host on aws
embed in website

Example:
https://github.com/mahdjourOussama/python-learning/tree/master/chatbot-rag
