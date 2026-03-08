import sys
from pathlib import Path

# Принудительно добавляем пути
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import config as cfg
except ImportError:
    print(f"❌ Ошибка: не удалось импортировать config")
    print(f"Путь к проекту: {project_root}")
    sys.exit(1)

import numpy as np
from src.data_models import DocumentData

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers не установлен")

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers не установлен")
        self.model_name = model_name
        print(f"🔄 Загрузка {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Готово, размерность: {self.model.get_sentence_embedding_dimension()}")

    def encode(self, texts: list) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype('float32')

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

def main():
    doc_path = cfg.PROCESSED_DIR / "document.json"
    if not doc_path.exists():
        print("❌ Файл не найден")
        return
    doc = DocumentData.load(doc_path)
    texts = doc.get_all_texts()
    print(f"📄 Найдено {len(texts)} текстов")
    if not texts:
        return
    generator = EmbeddingGenerator()
    embeddings = generator.encode(texts)
    np.save(cfg.DATA_DIR / "text_embeddings.npy", embeddings)
    import json
    meta = [{"page_num": p.page_num, "text": p.text[:200]} for p in doc.pages if p.text.strip()]
    with open(cfg.DATA_DIR / "embedding_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Готово")

if __name__ == "__main__":
    main()