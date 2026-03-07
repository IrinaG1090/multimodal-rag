import os
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
INDEX_DIR = DATA_DIR / "index"

# Создаем папки
for dir_path in [RAW_DIR, PROCESSED_DIR, IMAGES_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Параметры
PDF_DPI = 150
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
RERANK_TOP_K = 10

# Модели (твой API-ключ OpenAI должен быть в .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

# Параметры эмбеддингов
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # локальная модель для эмбеддингов
EMBEDDING_DIM = 384  # размерность для all-MiniLM-L6-v2

# FAISS параметры
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = INDEX_DIR / "index_metadata.json"

print("[OK] Конфигурация загружена")