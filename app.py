import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("Smart Document Q&A")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings)

qa_model = pipeline("question-answering")

query = st.text_input("Ask your question")

if query:
    docs = db.similarity_search(query, k=2)
    context = " ".join([d.page_content for d in docs])

    result = qa_model(question=query, context=context)
    st.write(result["answer"])

