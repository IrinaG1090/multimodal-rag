# 📚 Multimodal RAG — Поиск по документам с изображениями

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.10-green)](https://github.com/facebookresearch/faiss)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-orange)](https://openrouter.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 О проекте

**Multimodal RAG** — это система вопросно-ответного поиска по документам, которая понимает не только текст, но и изображения. Проект является седьмым этапом большого плана по созданию production-ready AI систем.

### ✨ Возможности

- 📄 **Парсинг PDF** — извлечение текста и изображений из PDF-файлов
- 🖼️ **Captioning изображений** — автоматическое описание картинок через OpenRouter API
- 🔍 **Мультимодальный поиск** — единый FAISS индекс для текста и описаний изображений
- 💬 **Поиск по запросу** — находит релевантные страницы и картинки
- 🐳 **Полная локальность** — все компоненты работают на вашем компьютере

## 🛠️ Технологический стек

| Компонент | Технология |
|-----------|------------|
| **Язык** | Python 3.12 |
| **Парсинг PDF** | PyMuPDF |
| **Векторная БД** | FAISS |
| **Эмбеддинги** | sentence-transformers (all-MiniLM-L6-v2) |
| **Captioning** | OpenRouter API (gemini-2.5-flash) |
| **Конфигурация** | python-dotenv |

## 📋 Предварительные требования

- Python 3.12+
- OpenRouter API ключ (получить на [openrouter.ai/keys](https://openrouter.ai/keys))

## 🚀 Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/IrinaG1090/multimodal-rag.git
cd multimodal-rag
```
### 2. Создать виртуальное окружение

```bash
python -m venv venv_mmrag
source venv_mmrag/bin/activate  # для Linux/Mac
.\venv_mmrag\Scripts\Activate.ps1   # для Windows
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Настроить переменные окружения
Создать файл .env из примера:

```bash
cp .env.example .env
```

Добавьте ваш OpenRouter ключ в файл .env:
OPENROUTER_API_KEY=ваш_ключ_сюда

### 5. Положите PDF в папку данных
Поместите ваш PDF-файл в папку data/raw/

### 6. Запустите пайплайн

```bash
# Шаг 1: Парсинг PDF (извлечение текста и картинок)
python src/document_parser.py
```
```bash
# Шаг 2: Обработка метаданных
python src/process_metadata.py
```
```bash
# Шаг 3: Генерация описаний изображений (через OpenRouter)
python src/openrouter_captioner.py
```
```bash
# Шаг 4: Создание мультимодального индекса
python src/index_images.py
```
```bash
# Шаг 5: Поиск
python src/search.py
```

## 📁 Структура проекта

multimodal-rag/
├── src/
│   ├── config.py                 # Настройки и пути
│   ├── data_models.py             # Классы данных
│   ├── document_parser.py         # Парсинг PDF
│   ├── process_metadata.py        # Обработка метаданных
│   ├── embeddings.py               # Генерация эмбеддингов
│   ├── indexer.py                  # FAISS индекс
│   ├── index_images.py             # Мультимодальный индекс
│   ├── openrouter_captioner.py     # Описания через OpenRouter
│   └── search.py                   # Поиск по индексу
├── data/
│   ├── raw/                        # PDF файлы
│   ├── processed/                   # Результаты парсинга
│   │   └── images/                  # Извлечённые изображения
│   └── index/                       # FAISS индексы
├── .env.example                     # Пример переменных
├── .gitignore                        # Игнорируемые файлы
├── requirements.txt                  # Зависимости
└── README.md                         # Документация

## 🧪 Примеры запросов
После индексации можно искать:

```bash
python src/search.py
```

Примеры запросов:

"Что изображено на графиках?"

"Найди описания диаграмм"

"Покажи страницы с изображениями"

"условия" — поиск по тексту

## 📊 Результаты
Параметр	        Значение

Текстовых чанков	26
Описаний картинок	58
Всего векторов	    84
Размерность	        384

## 🗺️ Roadmap

Парсинг PDF (текст + картинки)

Captioning через OpenRouter

Мультимодальный FAISS индекс

Поиск по тексту и изображениям

Веб-интерфейс на Streamlit

Поддержка нескольких PDF

Гибридный поиск (BM25 + векторный)

## 🤝 Вклад в проект
Буду рада любым предложениям и улучшениям! Создавайте issue или отправляйте pull request.

## 📄 Лицензия
Проект распространяется под лицензией MIT. Подробнее в файле LICENSE.

## 🙏 Благодарности
OpenRouter за доступ к моделям

Facebook Research за FAISS

Hugging Face за sentence-transformers

## Сделано с ❤️