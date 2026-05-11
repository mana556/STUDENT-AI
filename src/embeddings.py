import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings


def _load_dotenv():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_embeddings():
    _load_dotenv()
    # Use local embeddings to avoid API rate limits
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )