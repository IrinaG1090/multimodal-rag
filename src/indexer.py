# src/indexer.py
import sys
import json
import numpy as np
import faiss
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config as cfg


class FAISSIndexer:
    """Класс для работы с FAISS индексом"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.metadata = []

    def build_index(self, embeddings: np.ndarray, metadata: list):
        """
        Строит FAISS индекс из эмбеддингов.
        Использует L2 расстояние (можно заменить на косинусное через нормализацию).
        """
        print(f"🔄 Построение FAISS индекса...")
        print(f"   • Размерность: {self.dimension}")
        print(f"   • Количество векторов: {len(embeddings)}")

        # Создаем индекс L2
        self.index = faiss.IndexFlatL2(self.dimension)

        # Добавляем векторы
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata

        print(f"✅ Индекс построен. Всего векторов в индексе: {self.index.ntotal}")

    def save(self, index_path: Path, metadata_path: Path):
        """Сохраняет индекс и метаданные"""
        if self.index is None:
            raise ValueError("Нет индекса для сохранения")

        # Сохраняем FAISS индекс
        faiss.write_index(self.index, str(index_path))
        print(f"✅ FAISS индекс сохранен: {index_path}")

        # Сохраняем метаданные
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ Метаданные индекса сохранены: {metadata_path}")

    def load(self, index_path: Path, metadata_path: Path):
        """Загружает индекс и метаданные"""
        if not index_path.exists():
            raise FileNotFoundError(f"Индекс не найден: {index_path}")

        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"✅ Индекс загружен: {index_path}")
        print(f"   • Размерность: {self.dimension}")
        print(f"   • Векторов: {self.index.ntotal}")
        print(f"   • Записей метаданных: {len(self.metadata)}")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Ищет k ближайших соседей для запроса.
        Возвращает индексы, расстояния и метаданные.
        """
        if self.index is None:
            raise ValueError("Индекс не загружен")

        distances, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "rank": i + 1,
                    "index": int(idx),
                    "distance": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })

        return results

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Синоним для search, возвращает в формате, удобном для LangChain-подобных систем.
        """
        return self.search(query_embedding, k)


def create_index_from_files():
    """Создает индекс из ранее сохраненных эмбеддингов и метаданных"""
    # Пути к файлам
    embeddings_path = cfg.DATA_DIR / "text_embeddings.npy"
    metadata_path = cfg.DATA_DIR / "embedding_metadata.json"

    if not embeddings_path.exists():
        print(f"❌ Файл эмбеддингов не найден: {embeddings_path}")
        print("   Сначала запусти embeddings.py")
        return False

    if not metadata_path.exists():
        print(f"❌ Файл метаданных не найден: {metadata_path}")
        return False

    # Загружаем данные
    print(f"📂 Загрузка эмбеддингов из: {embeddings_path}")
    embeddings = np.load(embeddings_path)

    print(f"📂 Загрузка метаданных из: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"📊 Данные загружены:")
    print(f"   • Форма эмбеддингов: {embeddings.shape}")
    print(f"   • Количество метаданных: {len(metadata)}")

    # Проверка соответствия
    if len(embeddings) != len(metadata):
        print(f"⚠️  Предупреждение: количество эмбеддингов ({len(embeddings)}) "
              f"не совпадает с количеством метаданных ({len(metadata)})")

    # Создаем индекс
    indexer = FAISSIndexer(dimension=embeddings.shape[1])
    indexer.build_index(embeddings, metadata)

    # Сохраняем индекс
    index_path = cfg.INDEX_DIR / "faiss_index.bin"
    meta_index_path = cfg.INDEX_DIR / "index_metadata.json"

    indexer.save(index_path, meta_index_path)

    return True


def test_search():
    """Тестирует поиск по индексу"""
    indexer = FAISSIndexer(dimension=cfg.EMBEDDING_DIM)
    index_path = cfg.INDEX_DIR / "faiss_index.bin"
    meta_path = cfg.INDEX_DIR / "index_metadata.json"

    if not index_path.exists():
        print(f"❌ Индекс не найден. Сначала создай индекс.")
        return

    # Загружаем индекс
    indexer.load(index_path, meta_path)

    # Создаем тестовый запрос (просто первый вектор из индекса)
    test_vector = np.random.randn(cfg.EMBEDDING_DIM).astype('float32')

    # Ищем
    results = indexer.search(test_vector, k=3)

    print("\n🔍 Результаты тестового поиска:")
    for r in results:
        print(f"\n--- Ранг {r['rank']} (расстояние: {r['distance']:.4f}) ---")
        print(f"Страница: {r['metadata']['page_num']}")
        print(f"Текст: {r['metadata']['text'][:150]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 FAISS Index Builder")
    print("=" * 60)

    if create_index_from_files():
        print("\n" + "=" * 60)
        print("✅ Индекс успешно создан!")
        print("=" * 60)

        # Опционально: тестовый поиск
        test_choice = input("\n🔄 Запустить тестовый поиск? (y/n): ").strip().lower()
        if test_choice == 'y':
            test_search()
    else:
        print("\n❌ Не удалось создать индекс")