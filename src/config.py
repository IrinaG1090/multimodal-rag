import os
from pathlib import Path

# config.py лежит в src, BASE_DIR должен указывать на корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent  # поднимаемся на два уровня

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

# Параметры эмбеддингов
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# FAISS параметры
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = INDEX_DIR / "index_metadata.json"

print(f"[OK] Конфигурация загружена. BASE_DIR: {BASE_DIR}")
print(f"[OK] DATA_DIR: {DATA_DIR}")
print(f"[OK] PROCESSED_DIR: {PROCESSED_DIR}")