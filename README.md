# genai-document-qna
# Smart Document Q&A System (GenAI Mini Project)

AI-powered application that allows users to upload PDF documents and ask questions using semantic search and transformer models.
# Live Demo: https://genai-document-qna-nn8ddfbkynofpgpndbs8db.streamlit.app/

## Features
- PDF text extraction
- Embedding generation using Sentence Transformers
- FAISS vector database
- Question answering using HuggingFace models
- Streamlit UI

## Tech Stack
- Python
- LangChain
- FAISS
- HuggingFace Transformers
- Streamlit

## How it Works
1. PDF is parsed and split into chunks
2. Text converted into embeddings
3. Stored in FAISS vector DB
4. User question retrieves relevant chunks
5. Transformer model generates answer

## Run Instructions

```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
