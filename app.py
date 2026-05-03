import streamlit as st
from src.loader import load_pdf
from src.chunking import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag_pipeline import generate_answer

st.title("🎓 AI Student Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = load_pdf("data/temp.pdf")
    chunks = split_documents(docs)

    embeddings = get_embeddings()
    db = create_vector_store(chunks, embeddings)
    retriever = get_retriever(db)
    llm = get_llm()

    query = st.text_input("Ask a question")

    if query:
        answer, sources = generate_answer(query, retriever, llm)
        st.write("### Answer")
        st.write(answer)

        st.write("### Sources")
        for doc in sources:
            st.write(doc.metadata)