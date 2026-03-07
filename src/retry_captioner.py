# src/retry_captioner.py
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


def is_error_caption(caption: str) -> bool:
    """Проверяет, является ли описание ошибочным."""
    if not caption:
        return True
    error_indicators = ["[ОШИБКА", "ERROR", "429", "quota", "limit", "geolocation", "not supported"]
    caption_lower = caption.lower()
    return any(indicator.lower() in caption_lower for indicator in error_indicators)


def generate_caption(image_path: Path) -> str:
    """Генерирует описание изображения через Gemini."""
    try:
        img = Image.open(image_path)
        prompt = (
            "Опиши это изображение максимально подробно на русском языке. "
            "Если это график, диаграмма или схема, объясни, что он показывает. "
            "Если это таблица, перечисли все данные в структурированном виде. "
            "Если это просто картинка или фото, детально опиши, что на ней изображено."
        )
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return f"[ОШИБКА: {e}]"


def main():
    """Перегенерирует описания только для ошибочных изображений."""
    print("=" * 60)
    print("🔄 Повторная генерация описаний для ошибочных изображений")
    print("=" * 60)

    doc_path = cfg.PROCESSED_DIR / "document.json"
    if not doc_path.exists():
        print(f"❌ Файл {doc_path} не найден")
        return

    doc = DocumentData.load(doc_path)
    all_images = doc.get_all_images()
    print(f"📸 Всего изображений: {len(all_images)}")

    # Собираем ошибочные
    error_images = []
    for img in all_images:
        if is_error_caption(img.caption):
            error_images.append(img)

    print(f"⚠️  Найдено ошибочных описаний: {len(error_images)}")

    if not error_images:
        print("✅ Ошибочных описаний нет. Всё готово!")
        return

    # Перегенерируем
    success_count = 0
    for i, img in enumerate(error_images, 1):
        print(f"\n[{i}/{len(error_images)}] 🖼️  Обрабатываю: {img.path.name}")
        
        new_caption = generate_caption(img.path)
        img.caption = new_caption

        # Сохраняем в отдельный файл
        txt_path = img.path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_caption)

        if not is_error_caption(new_caption):
            success_count += 1
            print(f"   ✅ Успешно")
        else:
            print(f"   ⚠️  Снова ошибка")

        # Пауза для соблюдения лимитов
        time.sleep(2)

    # Сохраняем обновленный документ
    doc.save(doc_path)

    print("\n" + "=" * 60)
    print(f"✅ Готово!")
    print(f"   • Обработано изображений: {len(error_images)}")
    print(f"   • Успешно исправлено: {success_count}")
    print(f"   • Осталось с ошибками: {len(error_images) - success_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()