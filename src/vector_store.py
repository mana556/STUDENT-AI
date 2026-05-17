import os
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
import logging


def create_vector_store(docs, embeddings, store_type="faiss", collection_name="documents", append=False, persist_path=None):
    """
    Create or update a vector store using FAISS or Chroma.

    Args:
        docs: List of documents to store
        embeddings: Embedding model or embedding function
        store_type: "faiss" or "chroma"
        collection_name: Name of the collection (for Chroma)
        append: If True, try to load existing store and add documents to it
        persist_path: Optional path to persist the store

    Returns:
        Vector store instance
    """
    store_type = store_type.lower()
    if store_type == "chroma":
        persist_dir = persist_path or "embeddings/chroma_db"
        os.makedirs(persist_dir, exist_ok=True)

        # If appending and a store exists, try to load and add documents
        try:
            if append and os.path.exists(persist_dir):
                db = load_vector_store(persist_dir, embeddings, store_type="chroma", collection_name=collection_name)
                if docs:
                    db.add_documents(docs)
                    db.persist()
                return db
        except Exception as e:
            logging.warning(f"Could not append to existing Chroma store: {e}")

        # Create a new Chroma store
        return Chroma.from_documents(
            docs,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )

    # Default to FAISS
    persist_dir = persist_path or "embeddings/faiss_index"
    os.makedirs(persist_dir, exist_ok=True)

    try:
        if append and os.path.exists(persist_dir):
            db = load_vector_store(persist_dir, embeddings, store_type="faiss")
            if docs:
                db.add_documents(docs)
                db.save_local(persist_dir)
            return db
    except Exception as e:
        logging.warning(f"Could not append to existing FAISS store: {e}")

    db = FAISS.from_documents(docs, embeddings)
    # persist FAISS on disk for later reuse
    try:
        db.save_local(persist_dir)
    except Exception:
        pass
    return db


def save_vector_store(db, path, store_type="faiss"):
    """Save vector store to disk."""
    if store_type.lower() == "chroma":
        db.persist()  # Chroma persists automatically
    else:
        db.save_local(path)


def load_vector_store(path, embeddings, store_type="faiss", collection_name="documents"):
    """Load vector store from disk."""
    if store_type.lower() == "chroma":
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=path
        )
    else:
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
