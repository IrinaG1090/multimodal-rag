# src/process_metadata.py
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from src.data_models import DocumentData, PageData, ImageData


def load_raw_metadata() -> dict:
    """Загружает сырые метаданные из JSON"""
    meta_path = cfg.PROCESSED_DIR / "pages_meta.json"
    if not meta_path.exists():
        print(f"❌ Файл {meta_path} не найден. Сначала запусти document_parser.py")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_document(raw_data: list, source_pdf: Path) -> DocumentData:
    """Конвертирует сырые данные в структурированный документ"""
    pages = []

    for raw_page in raw_data:
        page_num = raw_page["page_num"]
        text = raw_page["text"]
        images = []

        # Преобразуем информацию об изображениях
        for img_path_str in raw_page.get("images", []):
            img_path = Path(img_path_str)
            # Извлекаем индексы из имени файла
            # Формат: page_001_img_01.png
            parts = img_path.stem.split("_")
            img_index = int(parts[-1]) if len(parts) >= 4 else 0

            img_data = ImageData(
                path=img_path,
                page_num=page_num,
                image_index=img_index
            )
            images.append(img_data)

        page = PageData(
            page_num=page_num,
            text=text,
            images=images
        )
        pages.append(page)

    # Ищем исходный PDF
    pdf_files = list(cfg.RAW_DIR.glob("*.pdf"))
    source_pdf = pdf_files[0] if pdf_files else Path("unknown.pdf")

    return DocumentData(
        source_path=source_pdf,
        pages=pages
    )


def main():
    print("🔄 Преобразование метаданных в структурированные данные...")

    # Загружаем сырые данные
    raw_data = load_raw_metadata()

    # Создаем документ
    doc = convert_to_document(raw_data, cfg.RAW_DIR / "source.pdf")

    # Сохраняем структурированные данные
    output_path = cfg.PROCESSED_DIR / "document.json"
    doc.save(output_path)

    print(f"✅ Структурированные данные сохранены: {output_path}")
    print(f"📊 Статистика:")
    print(f"   • Страниц: {len(doc.pages)}")
    print(f"   • Страниц с текстом: {sum(1 for p in doc.pages if p.text.strip())}")
    print(f"   • Всего изображений: {len(doc.get_all_images())}")
    print(f"   • Изображений по страницам:")
    for page in doc.pages:
        if page.images:
            print(f"     - Страница {page.page_num}: {len(page.images)} изобр.")


if __name__ == "__main__":
    main()