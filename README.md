# 🏷️ Lenta Tech — Распознавание ценников с видеопотока робота

> Решение для хакатона **Lenta Tech Life Hack** — трек «Полка под контролем»

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7.3-green)](https://github.com/PaddlePaddle/PaddleOCR)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-orange)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)

## 📋 Описание задачи

Разработать систему компьютерного зрения для автоматического распознавания ценников с видеопотока робота, движущегося вдоль стеллажей магазина. На выходе — структурированный CSV-файл с данными всех обнаруженных ценников.

## 🏗️ Архитектура решения

```
Видео с робота → video_loader → detector → preprocessor → ocr_engine → tracker → output.csv
```

Пайплайн состоит из 6 модулей:

1. **video_loader.py** — извлечение кадров, CLAHE, фильтрация дублей
2. **detector.py** — YOLOv8 + HSV-маски по цвету ценников
3. **preprocessor.py** — кроп, поворот 90°, апскейл до 600px
4. **ocr_engine.py** — PaddleOCR + pyzbar + EDSR для QR
5. **tracker.py** — IoU-трекинг, дедупликация, majority vote
6. **app.py** — Streamlit веб-интерфейс

## 📁 Структура проекта

```
lenta-cv/
├── app.py              # Streamlit веб-интерфейс
├── video_loader.py     # Гзагрузка и предобработка видео
├── detector.py         # Гдетекция ценников
├── preprocessor.py     # подготовка кропов для OCR
├── ocr_engine.py       # OCR, QR, штрихкоды
├── tracker.py          # трекинг и дедупликация
├── config.py           # Единый конфиг всего проекта
├── debug_ocr.py        # Отладка OCR на одном кропе
├── debug_qr.py         # Отладка QR-детектора
├── requirements.txt    # Зависимости Python
├── packages.txt        # Системные пакеты (для HF Spaces)
├── models/             # Папка для моделей (EDSR_x4.pb)
└── output/             # Папка для результатов CSV
└── gt_loader.py        # Проверка с эталонной табилцей
```

## 🔧 Установка

### Требования
- Python 3.10–3.12, Windows 10/11 x64 или Linux
- RAM: минимум 4 GB, место на диске: ~3 GB

### Быстрый старт

```bash
# 1. Создать окружение
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux

# 2. Обновить pip
python -m pip install --upgrade pip wheel setuptools

# 3. Установить numpy первым (важно!)
pip install "numpy>=1.26.0,<2.0.0"

# 4. Установить OpenCV
pip install opencv-python-headless==4.10.0.84

# 5. Установить PaddlePaddle
pip install paddlepaddle==2.6.2 --only-binary=:all:

# 6. Установить остальное
pip install -r requirements.txt
```

### zbar для штрихкодов

**Windows:** скачать `libzbar-64.dll` → https://github.com/NaturalHistoryMuseum/pyzbar/releases → положить в папку проекта

**Linux:**
```bash
sudo apt-get install libzbar0
```

### EDSR модель для QR (опционально, ~20MB)

```bash
wget -O models/EDSR_x4.pb https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb
```

## 🚀 Запуск

### Веб-интерфейс

```bash
streamlit run app.py
```

Браузер откроется на `http://localhost:8501`. Загрузи видео → нажми «Запустить» → скачай CSV.

### CLI

```bash
# Детекция + кропирование
python preprocessor.py videos/robot.mp4

# OCR всех кропов
python ocr_engine.py debug_crops/robot/

# Трекинг и экспорт CSV
python tracker.py debug_crops/robot/ocr_results.json
```

### Отладка

```bash
python debug_ocr.py debug_crops/robot/crop_0021_orange.jpg
python debug_qr.py debug_crops/robot/crop_0021_orange.jpg
python detector.py videos/robot.mp4
```

## 📊 Выходной формат CSV

| Поле | Описание |
|------|----------|
| `filename` | Имя видеофайла |
| `product_name` | Наименование товара |
| `price_default` | Цена без карты |
| `price_card` | Цена по карте лояльности |
| `price_discount` | Цена по акции |
| `barcode` | Штрихкод (EAN-13) |
| `discount_amount` | Размер скидки (-36%) |
| `id_sku` | Артикул товара |
| `print_datetime` | Дата и время печати |
| `color` | Цвет ценника |
| `frame_timestamp` | Время в видео (мс) |
| `x_min/y_min/x_max/y_max` | Координаты bbox |
| `qr_code_barcode` | Штрихкод из QR |
| `price1_qr...price4_qr` | Цены из QR |
| `action_price_qr` | Акционная цена из QR |

## ⚙️ Настройка (config.py)

```python
LOADER_CONFIG = VideoLoaderConfig(fps_sample=2.0)
DETECTOR_CONFIG = DetectorConfig(confidence=0.35)
PREPROCESSOR_CONFIG = PreprocessorConfig(min_ocr_width=200)
OCR_CONFIG = OCRConfig(lang="ch")
TRACKER_CONFIG = TrackerConfig(iou_threshold=0.3)
```

## ⚠️ Ограничения и идеи по масштабированию

**Текущие ограничения:**
- QR-коды плохо читаются при движении камеры (motion blur)
- Кириллица в мелком тексте иногда искажается моделью ch
- Ценники под стеклянными ограждениями снижают точность

**Идеи по масштабированию:**
- Дообучить YOLOv8 на ценниках Ленты через псевдо-лейблинг
- Фильтрация кадров по резкости — использовать только стоп-кадры для QR
- Оптимизация под RKNN int8 для развёртывания на борту робота
- Параллельная обработка нескольких потоков для ускорения OCR

## 📦 Зависимости

| Пакет | Версия | Назначение |
|-------|--------|-----------|
| `paddleocr` | 2.7.3 | OCR текста |
| `paddlepaddle` | 2.6.2 | Движок PaddleOCR |
| `ultralytics` | ≥8.2 | YOLOv8 детектор |
| `opencv-python-headless` | 4.10 | Обработка изображений |
| `pyzbar` | ≥0.1.9 | Штрихкоды и QR |
| `streamlit` | ≥1.28 | Веб-интерфейс |
| `pandas` | ≥2.0 | Экспорт CSV |
