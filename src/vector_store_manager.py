"""
Vector Store Manager - Utility functions for managing different vector stores.
Supports FAISS and Chroma DB.
"""

import os
import shutil
from pathlib import Path


def clear_vector_store(store_type="faiss"):
    """
    Clear a vector store by deleting its directory.
    
    Args:
        store_type: "faiss" or "chroma"
    """
    if store_type.lower() == "chroma":
        path = "embeddings/chroma_db"
    else:
        path = "embeddings/faiss_index"
    
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Cleared {store_type.upper()} vector store at {path}")
        return True
    return False


def get_store_size(store_type="faiss"):
    """Get the size of a vector store in MB."""
    if store_type.lower() == "chroma":
        path = "embeddings/chroma_db"
    else:
        path = "embeddings/faiss_index"
    
    if not os.path.exists(path):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)  # Convert to MB


def list_stored_documents(store_type="faiss"):
    """List information about stored documents."""
    path = "embeddings/chroma_db" if store_type.lower() == "chroma" else "embeddings/faiss_index"
    
    if not os.path.exists(path):
        return f"No {store_type.upper()} store found at {path}"
    
    size_mb = get_store_size(store_type)
    file_count = sum([len(files) for _, _, files in os.walk(path)])
    
    return {
        "store_type": store_type.upper(),
        "path": path,
        "size_mb": f"{size_mb:.2f}",
        "file_count": file_count,
        "exists": True
    }


def get_store_stats():
    """Get statistics for both vector stores."""
    return {
        "faiss": list_stored_documents("faiss"),
        "chroma": list_stored_documents("chroma"),
    }


def migrate_store(from_type, to_type, docs, embeddings):
    """
    Migrate documents from one vector store type to another.
    
    Args:
        from_type: Source store type ("faiss" or "chroma")
        to_type: Target store type ("faiss" or "chroma")
        docs: List of documents to migrate
        embeddings: Embedding model
    
    Returns:
        New vector store instance
    """
    from src.vector_store import create_vector_store
    
    print(f"Migrating from {from_type.upper()} to {to_type.upper()}...")
    
    # Create new store
    new_store = create_vector_store(docs, embeddings, store_type=to_type)
    
    # Clear old store if different
    if from_type != to_type:
        clear_vector_store(from_type)
    
    print(f"Migration complete!")
    return new_store
