# src/data_models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


@dataclass
class ImageData:
    """Данные об изображении, извлеченном из PDF"""
    path: Path
    page_num: int
    image_index: int
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "page_num": self.page_num,
            "image_index": self.image_index,
            "caption": self.caption,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageData':
        return cls(
            path=Path(data["path"]),
            page_num=data["page_num"],
            image_index=data["image_index"],
            caption=data.get("caption"),
            metadata=data.get("metadata", {})
        )


@dataclass
class PageData:
    """Данные одной страницы PDF"""
    page_num: int
    text: str
    images: List[ImageData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "page_num": self.page_num,
            "text": self.text,
            "images": [img.to_dict() for img in self.images],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PageData':
        return cls(
            page_num=data["page_num"],
            text=data["text"],
            images=[ImageData.from_dict(img) for img in data.get("images", [])],
            metadata=data.get("metadata", {})
        )


@dataclass
class DocumentData:
    """Данные всего документа"""
    source_path: Path
    pages: List[PageData] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source_path": str(self.source_path),
            "pages": [page.to_dict() for page in self.pages],
            "processed_at": self.processed_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentData':
        return cls(
            source_path=Path(data["source_path"]),
            pages=[PageData.from_dict(page) for page in data["pages"]],
            processed_at=datetime.fromisoformat(data["processed_at"]),
            metadata=data.get("metadata", {})
        )

    def save(self, path: Path) -> None:
        """Сохраняет документ в JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'DocumentData':
        """Загружает документ из JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_all_texts(self) -> List[str]:
        """Возвращает список всех текстов страниц"""
        return [page.text for page in self.pages if page.text.strip()]

    def get_all_images(self) -> List[ImageData]:
        """Возвращает список всех изображений"""
        return [img for page in self.pages for img in page.images]

    def get_texts_by_page(self, page_nums: List[int]) -> List[str]:
        """Возвращает тексты только для указанных страниц"""
        return [page.text for page in self.pages if page.page_num in page_nums]