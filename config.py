import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "<your-groq-api-key>")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector Store Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # "faiss" or "chroma"
VECTOR_DB_PATH = "embeddings/faiss_index"
CHROMA_DB_PATH = "embeddings/chroma_db"

# Store type info
STORE_INFO = {
    "faiss": {
        "name": "FAISS",
        "description": "Fast, memory-based vector store. Good for quick searches.",
        "path": VECTOR_DB_PATH,
        "persistent": False
    },
    "chroma": {
        "name": "Chroma",
        "description": "Persistent, lightweight vector store. Data saved to disk.",
        "path": CHROMA_DB_PATH,
        "persistent": True
    }
}
