**A chatbot using Retrieval-Augmented Generation (RAG) to answer questions or provide information based on provided documents.**

## Overview

This project is a document-aware chatbot powered by Retrieval-Augmented Generation (RAG), designed to provide intelligent, context-aware responses strictly within the scope of user-uploaded documents. It allows users to upload a variety of documents, from which the chatbot extracts and embeds text into a vector database for efficient semantic search and retrieval. 

Once documents are uploaded, users can engage in a multi-turn conversation with the chatbot. Leveraging the retained context of previous messages, the chatbot is capable of maintaining coherent and relevant dialogue throughout the session. 

Importantly, the chatbot is domain-constrained. It strictly answers questions related to the uploaded documents and does not entertain queries outside their scope. When presented with questions beyond the domain of the documents or when it lacks sufficient information, the chatbot explicitly states that it does not know the answer. 

## Demo

The chatbot is deployed in Streamlit Community Cloud. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chat-bot.streamlit.app/)

## Technologies Used

- 🐍 **Programming Language:** Python 3.13.0 
- 🔗 **LLM Orchestration:** LangChain, LangGraph 
- 🧬 **Embeddings Model:** `text-embedding-004` through the Gemini API
- 🔍 **Vector Database:** FAISS 
- 🌐 **Frontend Interface:** Streamlit 
- 🧠 **LLM Integration:** Google AI Studio  

## Installation

Clone the repository.
```
git clone https://github.com/prashalruchiranga/rag-chatbot.git
cd rag-chatbot
```
Create a virtual environment and activate. Ensure you are using **Python 3.13.0 or above**, as this project has not been tested with earlier versions.
```
uv venv --python=python3.13
source .venv/bin/activate
```
Install development dependencies.
```
uv sync
```
Run the chatbot.
```
python -m streamlit run main.py
```

## License

Licensed under MIT. See the [LICENSE](https://github.com/prashalruchiranga/rag-chatbot/blob/main/LICENSE).
