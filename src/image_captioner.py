# src/image_captioner.py
import sys
from pathlib import Path
from PIL import Image
import base64
import time
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
import config as cfg
from src.data_models import DocumentData

# Настройка OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY не найден в .env файле")
    print("📌 Получи ключ на https://openrouter.ai/keys")
    sys.exit(1)

# Конфигурация API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.5-flash"  # или "meta-llama/llama-3.3-70b-instruct" для текста
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",  # для статистики
    "X-Title": "Multimodal RAG Project"
}


def encode_image(image_path):
    """Кодирует изображение в base64 для отправки в API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_caption(image_path: Path) -> str:
    """Генерирует описание изображения через OpenRouter API."""
    print(f"🖼️  Генерирую описание для: {image_path.name}")
    try:
        base64_image = encode_image(image_path)

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Опиши это изображение максимально подробно на русском языке. "
                                "Если это график, диаграмма или схема, объясни, что он показывает. "
                                "Если это таблица, перечисли все данные в структурированном виде. "
                                "Если это просто картинка или фото, детально опиши, что на ней изображено."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.2
        }

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Ошибка API: {e}")
        return f"[ОШИБКА API: {e}]"
    except Exception as e:
        print(f"   ❌ Неизвестная ошибка: {e}")
        return f"[ОШИБКА: {e}]"


def main():
    """Основная функция: обрабатывает все изображения документа."""
    print("=" * 60)
    print("🔧 ЗАПУСК CAPTIONER ЧЕРЕЗ OPENROUTER")
    print("=" * 60)

    doc_path = cfg.PROCESSED_DIR / "document.json"
    if not doc_path.exists():
        print(f"❌ Файл {doc_path} не найден. Сначала запусти process_metadata.py")
        return

    doc = DocumentData.load(doc_path)
    all_images = doc.get_all_images()
    print(f"📸 Всего изображений в документе: {len(all_images)}")

    processed = 0
    skipped = 0

    for i, img_data in enumerate(all_images, 1):
        img_path = img_data.path
        if not img_path.exists():
            print(f"⚠️  Файл изображения не найден, пропускаем: {img_path}")
            skipped += 1
            continue

        # Проверяем, есть ли уже успешное описание
        if img_data.caption and "[ОШИБКА" not in img_data.caption and not img_data.caption.startswith("["):
            print(f"⏭️  Описание уже есть для {img_path.name}, пропускаем.")
            skipped += 1
            continue

        print(f"\n[{i}/{len(all_images)}] ", end="")
        caption = generate_caption(img_path)
        img_data.caption = caption
        processed += 1

        # Сохраняем описание в отдельный текстовый файл
        caption_file = img_path.with_suffix(".txt")
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)

        # Пауза для соблюдения лимитов
        time.sleep(3)

    # Сохраняем обновленный документ
    doc.save(doc_path)

    print("\n" + "=" * 60)
    print(f"✅ Обработка изображений завершена!")
    print(f"   • Обработано новых: {processed}")
    print(f"   • Пропущено (уже есть): {skipped}")
    print(f"   • Всего изображений: {len(all_images)}")
    print("=" * 60)


if __name__ == "__main__":
    main()