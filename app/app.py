import streamlit as st
import sys
import tempfile
from pathlib import Path

# ========== ПУТИ ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

for path in [str(PROJECT_ROOT), str(SRC_PATH)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import src.config as cfg
from src.embeddings import EmbeddingGenerator
import faiss
import json
from PIL import Image

# Импортируем модули для обработки PDF
from src.document_parser import parse_pdf, save_metadata
from src.process_metadata import convert_to_document
from src.openrouter_captioner import generate_caption
from src.index_images import create_combined_index

if __name__ == "__main__":
    import streamlit.web.bootstrap as bootstrap
    import sys
    bootstrap.run(str(__file__), False, [], {})

st.set_page_config(
    page_title="Multimodal RAG Search",
    page_icon="🔍",
    layout="wide"
)

st.title("📚 Multimodal RAG — Поиск по документам с изображениями")
st.markdown("Загрузите PDF и ищите информацию по тексту и картинкам")

# ==== ЗАГРУЗКА PDF ====
with st.expander("📤 Загрузить новый PDF", expanded=True):
    uploaded_file = st.file_uploader("Выберите PDF файл", type=['pdf'])
    
    if uploaded_file is not None:
        # Сохраняем загруженный файл во временную папку
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if st.button("🔄 Обработать PDF"):
            with st.spinner("Обработка документа... Это может занять несколько минут"):
                try:
                    # Шаг 1: Парсинг PDF
                    pages_data = parse_pdf(Path(tmp_path))
                    save_metadata(pages_data)
                    
                    # Шаг 2: Конвертация в DocumentData
                    doc = convert_to_document(pages_data, Path(tmp_path))
                    doc.save(cfg.PROCESSED_DIR / "document.json")
                    
                    # Шаг 3: Генерация описаний изображений
                    all_images = doc.get_all_images()
                    for img_data in all_images:
                        if not img_data.caption:
                            caption = generate_caption(img_data.path)
                            img_data.caption = caption
                            txt_path = img_data.path.with_suffix(".txt")
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(caption)
                    doc.save(cfg.PROCESSED_DIR / "document.json")
                    
                    # Шаг 4: Создание эмбеддингов и индекса
                    create_combined_index()
                    
                    st.success("✅ PDF успешно обработан и проиндексирован!")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке: {e}")
                finally:
                    # Удаляем временный файл
                    Path(tmp_path).unlink(missing_ok=True)

# ==== БОКОВАЯ ПАНЕЛЬ С ИНФОРМАЦИЕЙ ОБ ИНДЕКСЕ ====
with st.sidebar:
    st.header("ℹ️ О проекте")
    st.markdown("""
    **Multimodal RAG** — система поиска, которая понимает:
    - 📄 Текст страниц
    - 🖼️ Изображения (через их описания)
    
    **Технологии:**
    - FAISS для поиска
    - OpenRouter для описания картинок
    - Sentence-transformers для эмбеддингов
    """)
    
    try:
        index_path = cfg.INDEX_DIR / "multimodal_index.bin"
        meta_path = cfg.INDEX_DIR / "multimodal_metadata.json"
        
        if index_path.exists() and meta_path.exists():
            index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            st.success("✅ Индекс загружен")
            st.info(f"📊 Всего чанков: {len(metadata)}")
            st.info(f"📐 Размерность: {index.d}")
            
            text_count = sum(1 for item in metadata if item['type'] == 'text')
            image_count = sum(1 for item in metadata if item['type'] == 'image')
            st.info(f"📄 Текст: {text_count}")
            st.info(f"🖼️  Изображения: {image_count}")
            
            if st.button("🗑️ Очистить индекс"):
                index_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                st.rerun()
        else:
            st.warning("⚠️ Индекс не найден")
            st.info("Загрузите PDF выше для создания индекса")
            
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

# ==== ПОИСК ====
st.header("🔎 Поиск")

index_path = cfg.INDEX_DIR / "multimodal_index.bin"
if not index_path.exists():
    st.warning("Сначала загрузите и обработайте PDF")
    st.stop()

query = st.text_input("Введите ваш запрос:", placeholder="например: условия, графики, диаграммы")
k = st.slider("Количество результатов:", min_value=1, max_value=10, value=5)

if st.button("Найти", type="primary"):
    if not query.strip():
        st.warning("Введите запрос")
    else:
        with st.spinner("Ищем..."):
            try:
                generator = EmbeddingGenerator()
                index = faiss.read_index(str(index_path))
                
                with open(cfg.INDEX_DIR / "multimodal_metadata.json", "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                query_vec = generator.encode_single(query).reshape(1, -1)
                distances, indices = index.search(query_vec.astype('float32'), k)
                
                st.subheader(f"📌 Результаты поиска по запросу: '{query}'")
                
                found = False
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(metadata):
                        found = True
                        item = metadata[idx]
                        distance = distances[0][i]
                        
                        with st.container():
                            st.markdown(f"### [{i+1}] Результат (расстояние: {distance:.4f})")
                            
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                if item['type'] == 'image':
                                    st.markdown("🖼️ **Изображение**")
                                    img_path = Path(item.get('image_path', ''))
                                    if img_path.exists():
                                        img = Image.open(img_path)
                                        st.image(img, width=200)
                                    else:
                                        st.warning("Файл изображения не найден")
                                else:
                                    st.markdown("📄 **Текст**")
                            
                            with col2:
                                st.markdown(f"**Страница:** {item['page_num']}")
                                if item['type'] == 'text':
                                    st.markdown(f"**Текст:** {item['text'][:500]}...")
                                else:
                                    caption = item.get('caption', 'нет описания')
                                    st.markdown(f"**Описание:** {caption[:500]}...")
                            
                            st.divider()
                
                if not found:
                    st.warning("Ничего не найдено")
                    
            except Exception as e:
                st.error(f"❌ Ошибка при поиске: {e}")