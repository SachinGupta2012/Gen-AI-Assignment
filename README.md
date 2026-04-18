# Website RAG Chatbot Using Embeddings and Groq

## Project Overview

This project is an AI-powered chatbot that crawls any website, extracts its content, converts the text into vector embeddings, and answers user questions strictly based on that website’s content using Groq’s LLM API.

The chatbot follows a Retrieval-Augmented Generation (RAG) approach to ensure responses are grounded only in the provided website data.

## How It Works

1. User enters a website URL  
2. System crawls and extracts text  
   - Supports both static HTML and JavaScript-heavy websites  
3. Extracted text is split into chunks  
   - 1000 characters per chunk  
   - 200 characters overlap  
4. Text chunks are converted into embeddings using HuggingFace sentence-transformers  
5. Embeddings are stored in a FAISS vector database  
6. User asks a question  
7. Relevant chunks are retrieved using similarity search  
8. Groq LLM (Llama 3.3 70B) generates an answer strictly from the retrieved context  


## Installation

### Prerequisites

- Python 3.11 to 3.13  
- Google Chrome (required for JavaScript rendering)

### Setup

Create virtual environment
Install the libraries


## Running the Application

```bash
streamlit run app.py
```

##Steps:
Steps

Open your browser at http://localhost:8501

Enter your Groq API key in the sidebar

Enter a website URL and click Crawl & Index

Switch to the Chat Interface tab to ask questions

## Key Features

- **Vector Database**: FAISS (Facebook AI Similarity Search)  
- **Embeddings Model**: sentence-transformers/all-MiniLM-L6-v2 (local and free)  
- **LLM**: Groq API with Llama 3.3 70B  
  - High performance with ~500 tokens per second  
- **JavaScript Rendering**: Toggle support for dynamic websites  
- **Persistence**: Save and load FAISS index from disk  
- **Conversation Memory**: Retains the last 5 chat exchanges 



## Project Structure

```text
.
├── app.py                   # Main Streamlit application
├── src
│   ├── crawler.py           # Website crawling and extraction logic
│   ├── embeddings.py        # Text chunking and FAISS vector storage
│   └── chatbot.py           # RAG pipeline and Groq LLM integration
├── requirements.txt         # Python dependencies
└── screenshots/             # Optional screenshots for README


- For static websites (e.g. Wikipedia):  
  - Keep **JavaScript Rendering OFF**

- For dynamic websites (e.g. GeeksforGeeks):  
  - Enable **JavaScript Rendering**

- If the answer does not exist on the website, the chatbot returns exactly:

```text
The answer is not available on the provided website```



