import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")

# Open-Source Model Settings
# Make sure you have Ollama running locally (e.g., 'ollama run llama3')
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Hugging Face Local Embedding Model
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# LangSmith Tracing (Optional, but great for debugging)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "hybrid-rag-opensource")