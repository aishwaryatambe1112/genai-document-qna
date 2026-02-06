import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Document Q&A", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #f3f6ff, #eef1ff);
}

.big-title {
    font-size:56px;
    font-weight:900;
    text-align:center;
    color:white;
}

.sub {
    text-align:center;
    color:#cfcfe8;
    font-size:18px;
    margin-bottom:30px;
}

.answer-box {
    background:#1f1f2e;
    color:white;
    padding:20px;
    border-radius:12px;
    margin-top:10px;
    font-size:20px;
}

.question {
    font-weight:700;
    font-size:22px;
    color:#4cc9f0;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="big-title">📄 SMART DOCUMENT Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Retrieval Augmented Generative AI</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose document", type=["pdf"])

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_llm()

# ---------------- PROCESS PDF ----------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("vectorstore")

    st.sidebar.success("Document indexed!")

# ---------------- LOAD VECTORSTORE ----------------
if not os.path.exists("vectorstore"):
    st.info("Upload a PDF to start.")
    st.stop()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# ---------------- CHAT ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_input("💬 Ask your question")

if query:
    with st.spinner("Generating answer..."):

        docs = db.similarity_search(query, k=3)
        context = " ".join([d.page_content for d in docs])

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

        response = generator(prompt, max_length=300)
        answer = response[0]["generated_text"]

        st.session_state.chat.append((query, answer))

# ---------------- DISPLAY CHAT ----------------
for q, a in reversed(st.session_state.chat):
    st.markdown(f"<div class='question'>You: {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer-box'>{a}</div>", unsafe_allow_html=True)
