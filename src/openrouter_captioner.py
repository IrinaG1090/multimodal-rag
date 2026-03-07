# src/openrouter_captioner.py
import sys
from pathlib import Path
from PIL import Image
import base64
import time
from dotenv import load_dotenv
import os
import requests

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
import config as cfg
from src.data_models import DocumentData

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY не найден в .env файле")
    sys.exit(1)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Multimodal RAG Project"
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption(image_path: Path) -> str:
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
                                "Если это таблица, перечисли все данные в структурированном виде."
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

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return f"[ОШИБКА: {e}]"

def main():
    print("=" * 60)
    print("🔧 ОБРАБОТКА ВСЕХ ИЗОБРАЖЕНИЙ ЧЕРЕЗ OPENROUTER")
    print("=" * 60)

    doc_path = cfg.PROCESSED_DIR / "document.json"
    doc = DocumentData.load(doc_path)
    all_images = doc.get_all_images()
    print(f"📸 Всего изображений: {len(all_images)}")

    processed = 0
    for i, img_data in enumerate(all_images, 1):
        img_path = img_data.path
        
        # Пропускаем, если уже есть описание
        if img_data.caption and "[ОШИБКА" not in img_data.caption:
            print(f"⏭️  [{i}/{len(all_images)}] Уже есть: {img_path.name}")
            continue

        print(f"\n[{i}/{len(all_images)}] ", end="")
        caption = generate_caption(img_path)
        img_data.caption = caption
        processed += 1
        
        caption_file = img_path.with_suffix(".txt")
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)
        
        time.sleep(2)  # пауза для соблюдения лимитов

    doc.save(doc_path)
    print(f"\n✅ Обработано новых: {processed}")

if __name__ == "__main__":
    main()