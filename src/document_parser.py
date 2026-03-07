import json
import sys
from pathlib import Path

# Добавляем путь к корню проекта для импорта config
sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
import pymupdf  # Официальный импорт согласно документации [citation:1][citation:7]


def parse_pdf(pdf_path: Path) -> list:
    """
    Разбирает PDF на текст и изображения по страницам.
    Основано на официальной документации PyMuPDF [citation:1][citation:7].

    Args:
        pdf_path: Путь к PDF файлу

    Returns:
        Список словарей с данными каждой страницы
    """
    # Открываем документ согласно документации [citation:1]
    doc = pymupdf.open(str(pdf_path))
    pages_data = []

    print(f"[INFO] Обрабатываю документ: {pdf_path.name}")
    print(f"[INFO] Всего страниц: {len(doc)}")

    # Итерируем по страницам [citation:1]
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Извлекаем текст с помощью get_text() [citation:1]
        text = page.get_text()

        # Извлекаем изображения с помощью get_images() [citation:1]
        image_list = page.get_images(full=True)
        images = []

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]  # XREF изображения

            # Создаем Pixmap [citation:1]
            pix = pymupdf.Pixmap(doc, xref)

            # Обработка CMYK изображений [citation:1]
            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            # Генерируем имя файла
            img_filename = f"page_{page_num+1:03d}_img_{img_index:02d}.png"
            img_path = cfg.IMAGES_DIR / img_filename

            # Сохраняем изображение [citation:1]
            pix.save(str(img_path))
            pix = None  # Освобождаем память

            images.append(str(img_path))
            print(f"  → Сохранено изображение: {img_filename}")

        # Собираем данные страницы
        page_data = {
            "page_num": page_num + 1,
            "text": text.strip(),
            "images": images,
            "has_images": len(images) > 0,
            "has_text": len(text.strip()) > 0,
        }

        pages_data.append(page_data)
        print(f"✅ Страница {page_num+1}: текст {len(text)} симв., "
              f"изображений {len(images)}")

    doc.close()
    return pages_data


def save_metadata(pages_data: list) -> None:
    """Сохраняет метаданные страниц в JSON файл."""
    output_path = cfg.PROCESSED_DIR / "pages_meta.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Метаданные сохранены: {output_path}")
    print(f"✅ Всего страниц обработано: {len(pages_data)}")


def main():
    """Основная функция: находит PDF, парсит, сохраняет."""
    # Проверяем наличие PDF файлов
    pdf_files = list(cfg.RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ В папке {cfg.RAW_DIR} нет PDF файлов")
        print(f"📁 Пожалуйста, положите PDF в: {cfg.RAW_DIR}")
        sys.exit(1)

    # Берем первый PDF (или можно модифицировать для выбора)
    pdf_path = pdf_files[0]
    print(f"📄 Найден файл: {pdf_path.name}")

    # Парсим PDF
    pages_data = parse_pdf(pdf_path)

    # Сохраняем метаданные
    save_metadata(pages_data)


if __name__ == "__main__":
    main()