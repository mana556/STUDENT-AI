# Chroma DB Implementation Guide

## Overview

This project now supports two vector store backends:
- **FAISS**: Fast in-memory vector search (default)
- **Chroma DB**: Persistent, lightweight vector database

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

This includes:
- `chromadb`: Chroma vector database
- `langchain-chroma`: LangChain integration for Chroma

## Usage

### Select Vector Store in App

When you run the Streamlit app, you'll see a **Vector Store** selector in the sidebar:

```
Vector Store: [● FAISS  ○ Chroma]
```

- **FAISS**: Fast, memory-based. Resets when app restarts.
- **Chroma**: Persistent, data saved to `embeddings/chroma_db/`

### Upload and Index a PDF

1. Launch the app: `streamlit run app.py`
2. Choose your preferred vector store (FAISS or Chroma)
3. Upload a PDF file
4. The app automatically creates and indexes the document

## Implementation Details

### File Structure

```
src/
├── vector_store.py          # Core vector store logic (FAISS + Chroma)
├── vector_store_manager.py  # Utility functions for store management
└── [other files unchanged]

embeddings/
├── faiss_index/             # FAISS store (if used)
└── chroma_db/               # Chroma store (if used)
```

### Key Functions

#### `create_vector_store(docs, embeddings, store_type="faiss", collection_name="documents")`

Creates a vector store with documents.

```python
from src.vector_store import create_vector_store
from src.embeddings import get_embeddings

embeddings = get_embeddings()

# Using FAISS
db_faiss = create_vector_store(docs, embeddings, store_type="faiss")

# Using Chroma (persistent)
db_chroma = create_vector_store(docs, embeddings, store_type="chroma")
```

#### `load_vector_store(path, embeddings, store_type="faiss", collection_name="documents")`

Loads a previously saved vector store.

```python
from src.vector_store import load_vector_store

db = load_vector_store("embeddings/chroma_db", embeddings, store_type="chroma")
```

#### `save_vector_store(db, path, store_type="faiss")`

Saves a vector store to disk.

```python
from src.vector_store import save_vector_store

save_vector_store(db, "embeddings/chroma_db", store_type="chroma")
```

### Vector Store Manager

Use `vector_store_manager.py` for advanced operations:

```python
from src.vector_store_manager import get_store_stats, clear_vector_store, migrate_store

# Get statistics
stats = get_store_stats()
print(stats)

# Clear a store
clear_vector_store("chroma")

# Migrate documents to a different store
new_db = migrate_store("faiss", "chroma", docs, embeddings)
```

## Comparison

| Feature | FAISS | Chroma |
|---------|-------|--------|
| **Persistence** | No | Yes ✓ |
| **Speed** | Very Fast | Fast |
| **Memory** | Higher | Lower |
| **Reload** | Full reindex | Load from disk ✓ |
| **File Size** | Larger | Smaller |
| **Best For** | Quick demos | Production |

## Environment Variable

Set the default vector store type via environment variable:

```bash
export VECTOR_STORE_TYPE=chroma  # or "faiss"
```

Then update `config.py` to use it:

```python
from config import VECTOR_STORE_TYPE
db = create_vector_store(docs, embeddings, store_type=VECTOR_STORE_TYPE)
```

## Troubleshooting

### Chroma DB Issues

**Issue**: `ModuleNotFoundError: No module named 'chromadb'`

**Solution**: Install dependencies
```bash
pip install chromadb langchain-chroma
```

### Clear All Vector Stores

```bash
rm -rf embeddings/faiss_index
rm -rf embeddings/chroma_db
```

### Verify Store Contents

```python
from src.vector_store_manager import list_stored_documents

# Check FAISS
print(list_stored_documents("faiss"))

# Check Chroma
print(list_stored_documents("chroma"))
```

## Next Steps

- Switch between stores in the app sidebar
- Try Chroma for persistent storage
- Use `vector_store_manager.py` to manage multiple documents
- Set `VECTOR_STORE_TYPE=chroma` for production deployments
