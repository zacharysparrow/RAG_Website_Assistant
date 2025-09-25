# Retrieval-Augmented Generation (RAG) Website AI Agent

Agentic RAG chatbot made to answer questions about me and my work. 

## Built with
- Python
- Google Gemini
- LangChain
- Chroma DB
- FastAPI
- Streamlit
- Pydantic

## Getting Started
- Install all packages in requirements.txt
- make data/ directory for the chromadb
- make documents/ directory for the documents you want the RAG to read. Documents must be in a markdown format. You can convert PDFs to markdown with the help of some AI tools, such as those at [Datalab](https://www.datalab.to/playground).
- make a .env file containing your Google Gemini [API key](https://ai.google.dev/gemini-api/docs/api-key). For example, GEMINI_API_KEY="YOUR_KEY_HERE".
- For development, start the FastAPI with ```fastapi dev api.py``` and the streamlit app with ```streamlit run app.py```.

## Hosting
The streamlit frontend and FastAPI must be hosted seperately. I used render.com's free hobby tier to host the chatbot API, and the streamlit front end is hosted on streamlit community cloud, also for free. One of the benefits of using render to host the API is render's traditional server deployment, which allows for chat history management in memory. In contrast, a serverless deployment will require a persistent implementation of chat message history by maintaining e.g. a Postgres database.

## License
Distributed under the MIT License. See ```LICENSE``` for more information.

## TODO
add sldf paper when published
add boxes paper when on arxiv/published
fill out zach_info.md with answers to common interview questions
info on how the chatbot was made

