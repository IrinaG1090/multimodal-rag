# src/search.py
import sys
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config as cfg
import faiss
import json
import numpy as np
from src.embeddings import EmbeddingGenerator

def search(query: str, k: int = 5):
    """Поиск по мультимодальному индексу"""
    try:
        # Загружаем индекс
        index = faiss.read_index(str(cfg.INDEX_DIR / "multimodal_index.bin"))
        
        # Загружаем метаданные с правильной кодировкой UTF-8
        with open(cfg.INDEX_DIR / "multimodal_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Создаём эмбеддинг запроса
        generator = EmbeddingGenerator()
        query_vec = generator.encode_single(query).reshape(1, -1)
        
        # Поиск
        distances, indices = index.search(query_vec.astype('float32'), k)
        
        # Результаты
        print(f"\n🔍 Запрос: {query}\n")
        found = False
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(metadata):
                found = True
                item = metadata[idx]
                print(f"[{i+1}] (расстояние: {distances[0][i]:.4f})")
                print(f"   Тип: {item['type']}")
                print(f"   Страница: {item['page_num']}")
                if item['type'] == 'text':
                    print(f"   Текст: {item['text'][:150]}...")
                else:
                    print(f"   Картинка: {item.get('image_path', 'не указан')}")
                    print(f"   Описание: {item.get('caption', 'нет описания')[:150]}...")
                print()
        
        if not found:
            print("❌ Ничего не найдено")
        return metadata, indices, distances
        
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return [], [], []

if __name__ == "__main__":
    query = input("🔎 Введите запрос: ")
    search(query)