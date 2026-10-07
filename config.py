import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fungsi sederhana membaca .env tanpa dependensi tambahan
def load_dotenv_simple(env_path):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

# Muat file .env jika ada
load_dotenv_simple(os.path.join(BASE_DIR, ".env"))

# Data paths
DATA_PATH = os.path.join(BASE_DIR, "data", "berita_kesehatan_fix.csv")
DATA_WITH_REF_PATH = os.path.join(BASE_DIR, "data", "berita_kesehatan_with_reference.csv")
CSV_DELIMITER = "|"

# Vector DB & Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHROMA_PERSIST_DIR = os.path.join(OUTPUT_DIR, "index_db")
SUMMARIES_DIR = os.path.join(OUTPUT_DIR, "summaries")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Google Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.8-27b")

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

# Alibaba Cloud DashScope (Qwen Direct) configuration
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")

REFERENCE_PROVIDER = os.environ.get("REFERENCE_PROVIDER", "groq")

# Ollama Models configuration
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3:1.7b"       # Pilihan lain: "qwen3:0.6b"
DEFAULT_EMBED_MODEL = "nomic-embed-text" # Pilihan lain: "all-minilm:33m"

# Text Chunking parameters
CHUNK_SIZE = 500       # Jumlah karakter per chunk
CHUNK_OVERLAP = 50     # Overlap karakter antar chunk

# Experiment parameters
TOP_K_LIST = [3, 5, 10]  # Skenario nilai Top-K retrieval
TEMPERATURE = 0.3        # Suhu generasi LLM (rendah agar lebih faktual)

# Pastikan direktori output tersedia
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

