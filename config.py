import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "<your-groq-api-key>")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

VECTOR_DB_PATH = "embeddings/faiss_index"