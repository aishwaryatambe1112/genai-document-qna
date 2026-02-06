import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

st.set_page_config(page_title="Smart Document Q&A", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
<style>

body {
    background: linear-gradient(120deg, #f3f6ff, #eef1ff);
}

.big-title {
    font-size:56px !important;
    font-weight:900;
    text-align:center;
    color:#f5f5f5;
    margin-bottom:5px;
}

.sub {
    text-align:center;
    color:#5c5f7a;
    font-size:18px;
    margin-bottom:35px;
}

.answer-box {
    background:#1f1f2e;
    color:white;
    padding:22px;
    border-radius:14px;
    margin-top:10px;
    box-shadow:0px 6px 15px rgba(0,0,0,0.2);
    font-size:20px;
    animation: fadeIn 0.5s ease-in;
}

.question {
    font-weight:700;
    font-size:22px;
    color:#3a0ca3;
    margin-top:25px;
}

@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

.stTextInput>div>div>input {
    border-radius:12px;
    padding:14px;
    font-size:18px;
}

.sidebar .sidebar-content {
    background:#f7f8ff;
}

</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="big-title">📄 SMART DOCUMENT Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Ask questions from your documents using Generative AI</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.markdown("### 📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a document", type=["pdf"])

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

    st.sidebar.success("Document ready!")

# ---------- Load Vector DB ----------
if os.path.exists("vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
else:
    st.info("Upload a PDF to begin.")
    st.stop()

qa_model = pipeline("question-answering")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_input("💬 Ask your question")

if query:
    with st.spinner("Thinking..."):
        docs = db.similarity_search(query, k=2)
        context = " ".join([d.page_content for d in docs])

        result = qa_model(question=query, context=context)
        st.session_state.chat.append((query, result["answer"]))

# ---------- Chat History ----------
for q, a in reversed(st.session_state.chat):
    st.markdown(f"<div class='question'>You: {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer-box'>{a}</div>", unsafe_allow_html=True)
