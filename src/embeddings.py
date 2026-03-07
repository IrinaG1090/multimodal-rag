# src/embeddings.py
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from src.data_models import DocumentData

# Попробуем загрузить sentence-transformers, если не получится — дадим понятную ошибку
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers не установлен. Установи: pip install sentence-transformers")


class EmbeddingGenerator:
    """Класс для генерации эмбеддингов текста"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализирует модель эмбеддингов.
        По умолчанию используется легкая и быстрая модель all-MiniLM-L6-v2.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers не установлен")

        self.model_name = model_name
        print(f"🔄 Загрузка модели эмбеддингов: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Модель загружена. Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}")

    def encode(self, texts: list) -> np.ndarray:
        """
        Преобразует список текстов в эмбеддинги.
        """
        if not texts:
            return np.array([])

        print(f"🔄 Создание эмбеддингов для {len(texts)} текстов...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # L2 normalization отключена, будем использовать косинусное сходство
        )
        return embeddings.astype('float32')

    def encode_single(self, text: str) -> np.ndarray:
        """Преобразует один текст в эмбеддинг"""
        return self.encode([text])[0]


def main():
    """Основная функция для тестирования генерации эмбеддингов"""
    # Загружаем документ
    doc_path = cfg.PROCESSED_DIR / "document.json"
    if not doc_path.exists():
        print(f"❌ Файл {doc_path} не найден. Сначала запусти process_metadata.py")
        return

    doc = DocumentData.load(doc_path)

    # Получаем все тексты страниц
    texts = doc.get_all_texts()
    print(f"📄 Найдено {len(texts)} страниц с текстом")

    if not texts:
        print("⚠️ Нет текста для создания эмбеддингов")
        return

    # Создаем эмбеддинги
    generator = EmbeddingGenerator()
    embeddings = generator.encode(texts)

    # Сохраняем эмбеддинги
    output_path = cfg.DATA_DIR / "text_embeddings.npy"
    np.save(output_path, embeddings)

    # Сохраняем метаданные для эмбеддингов (соответствие индекс -> страница)
    text_metadata = [{"page_num": page.page_num, "text": page.text[:200]} for page in doc.pages if page.text.strip()]
    meta_output = cfg.DATA_DIR / "embedding_metadata.json"
    import json
    with open(meta_output, "w", encoding="utf-8") as f:
        json.dump(text_metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Эмбеддинги сохранены: {output_path}")
    print(f"📊 Размерность: {embeddings.shape}")
    print(f"✅ Метаданные сохранены: {meta_output}")


if __name__ == "__main__":
    main()