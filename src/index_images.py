# src/index_images.py
import sys
import json
import numpy as np
import faiss
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config as cfg
from src.data_models import DocumentData
from src.embeddings import EmbeddingGenerator


def load_image_captions(doc: DocumentData) -> tuple:
    """
    Извлекает все описания изображений из документа.
    Возвращает список текстов (caption) и список метаданных.
    """
    captions = []
    metadata = []

    all_images = doc.get_all_images()
    print(f"📸 Найдено изображений в документе: {len(all_images)}")

    for img_data in all_images:
        if img_data.caption and img_data.caption.strip():
            captions.append(img_data.caption)
            metadata.append({
                "type": "image",
                "page_num": img_data.page_num,
                "image_index": img_data.image_index,
                "image_path": str(img_data.path),
                "caption": img_data.caption[:200] + "..."  # превью для метаданных
            })

    print(f"📝 Из них с описаниями: {len(captions)}")
    return captions, metadata


def load_text_metadata() -> list:
    """Загружает существующие метаданные текстовых чанков."""
    meta_path = cfg.INDEX_DIR / "index_metadata.json"
    if not meta_path.exists():
        print(f"⚠️  Файл метаданных не найден: {meta_path}")
        return []
    
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_combined_index():
    """
    Создает объединенный индекс из текста страниц и описаний изображений.
    """
    print("=" * 60)
    print("🔧 Создание объединенного мультимодального индекса")
    print("=" * 60)

    # 1. Загружаем документ с описаниями
    doc_path = cfg.PROCESSED_DIR / "document.json"
    if not doc_path.exists():
        print(f"❌ Файл {doc_path} не найден")
        return False

    doc = DocumentData.load(doc_path)

    # 2. Получаем описания картинок
    image_captions, image_metadata = load_image_captions(doc)

    # 3. Загружаем существующие текстовые метаданные
    text_metadata = load_text_metadata()

    # 4. Объединяем все тексты и метаданные
    all_texts = []
    all_metadata = []

    # Добавляем текст страниц (если есть метаданные)
    if text_metadata:
        print(f"📄 Найдено текстовых чанков: {len(text_metadata)}")
        # Здесь нужно загрузить исходные тексты страниц из document.json
        for page in doc.pages:
            if page.text.strip():
                all_texts.append(page.text)
                all_metadata.append({
                    "type": "text",
                    "page_num": page.page_num,
                    "text": page.text[:200] + "..."
                })

    # Добавляем описания картинок
    all_texts.extend(image_captions)
    all_metadata.extend(image_metadata)

    print(f"\n📊 Всего чанков для индексации: {len(all_texts)}")
    print(f"   • Текст страниц: {len(all_texts) - len(image_captions)}")
    print(f"   • Описания картинок: {len(image_captions)}")

    if not all_texts:
        print("❌ Нет данных для индексации")
        return False

    # 5. Создаем эмбеддинги
    print("\n🔄 Создание эмбеддингов...")
    generator = EmbeddingGenerator()
    embeddings = generator.encode(all_texts)

    # 6. Создаем FAISS индекс
    print("\n🔄 Построение FAISS индекса...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    # 7. Сохраняем новый индекс и метаданные
    index_path = cfg.INDEX_DIR / "multimodal_index.bin"
    meta_path = cfg.INDEX_DIR / "multimodal_metadata.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Новый мультимодальный индекс сохранен:")
    print(f"   • Индекс: {index_path}")
    print(f"   • Метаданные: {meta_path}")
    print(f"   • Размерность: {embeddings.shape[1]}")
    print(f"   • Векторов: {len(embeddings)}")

    return True


def test_multimodal_search():
    """Тестирует поиск по новому мультимодальному индексу."""
    from src.embeddings import EmbeddingGenerator
    
    index_path = cfg.INDEX_DIR / "multimodal_index.bin"
    meta_path = cfg.INDEX_DIR / "multimodal_metadata.json"

    if not index_path.exists() or not meta_path.exists():
        print("❌ Индекс не найден. Сначала создай его.")
        return

    # Загружаем индекс и метаданные
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"\n🔍 Загружен индекс: {index_path}")
    print(f"   • Векторов: {index.ntotal}")
    print(f"   • Записей метаданных: {len(metadata)}")

    # Тестовый запрос
    generator = EmbeddingGenerator()
    
    test_queries = [
    "Что изображено на графиках в документе?",
    "Найди описания диаграмм и схем",
    "Покажи страницы с изображениями"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Запрос: {query}")
        print(f"{'='*60}")

        query_vec = generator.encode_single(query).reshape(1, -1)
        distances, indices = index.search(query_vec.astype('float32'), 3)

        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(metadata):
                item = metadata[idx]
                print(f"\n--- Результат {i+1} (расстояние: {distances[0][i]:.4f}) ---")
                print(f"Тип: {item['type']}")
                print(f"Страница: {item['page_num']}")
                if item['type'] == 'text':
                    print(f"Текст: {item['text']}")
                else:
                    print(f"Картинка: {item['image_path']}")
                    print(f"Описание: {item['caption'][:150]}...")


if __name__ == "__main__":
    if create_combined_index():
        print("\n" + "="*60)
        print("✅ Мультимодальный индекс успешно создан!")
        print("="*60)
        
        test_choice = input("\n🔄 Запустить тестовый поиск? (y/n): ").strip().lower()
        if test_choice == 'y':
            test_multimodal_search()