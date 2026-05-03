from langchain.vectorstores import FAISS

def create_vector_store(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

def save_vector_store(db, path):
    db.save_local(path)

def load_vector_store(path, embeddings):
    return FAISS.load_local(path, embeddings)