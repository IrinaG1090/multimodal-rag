# src/test_captioner.py
import sys
from pathlib import Path
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
import time

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
import config as cfg
from src.data_models import DocumentData

# Настройка Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY не найден в .env файле")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def generate_caption(image_path: Path) -> str:
    """Генерирует описание изображения через Gemini."""
    try:
        img = Image.open(image_path)
        prompt = (
            "Опиши это изображение максимально подробно на русском языке. "
            "Если это график, диаграмма или схема, объясни, что он показывает. "
            "Если это таблица, перечисли все данные."
        )
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return f"[ОШИБКА: {e}]"

def main():
    print("=" * 60)
    print("🔍 ТЕСТОВЫЙ ЗАПУСК: обработка 5 изображений")
    print("=" * 60)

    doc_path = cfg.PROCESSED_DIR / "document.json"
    doc = DocumentData.load(doc_path)
    
    # Берём первые 5 изображений
    all_images = doc.get_all_images()
    test_images = all_images[:5]
    
    print(f"📸 Всего изображений: {len(all_images)}")
    print(f"🧪 Тестовых изображений: {len(test_images)}")
    
    success = 0
    for i, img in enumerate(test_images, 1):
        print(f"\n[{i}/5] 🖼️  Обрабатываю: {img.path.name}")
        
        caption = generate_caption(img.path)
        img.caption = caption
        
        # Сохраняем
        txt_path = img.path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)
        
        if "ОШИБКА" not in caption:
            success += 1
            print(f"   ✅ Успешно ({len(caption)} симв.)")
        else:
            print(f"   ⚠️  Ошибка")
        
        time.sleep(12)  # пауза 12 секунд = 5 запросов в минуту
    
    # Сохраняем документ
    doc.save(doc_path)
    
    print("\n" + "=" * 60)
    print(f"✅ Тест завершён!")
    print(f"   • Успешно: {success}/5")
    print("=" * 60)

if __name__ == "__main__":
    main()