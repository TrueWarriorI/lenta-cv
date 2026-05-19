"""
ocr_engine.py — Глава 4: OCR и распознавание кодов
====================================================
Принимает кроп ценника (BGR numpy array) и возвращает
структурированный словарь со всеми полями для CSV.

Пайплайн:
  1. PaddleOCR — читает весь текст с кропа
  2. pyzbar   — декодирует штрихкоды
  3. QR-декодер — декодирует QR и парсит поля
  4. Парсер   — сопоставляет блоки текста с полями ценника
               по позиции, размеру, regex-паттернам
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Конфиг
# ─────────────────────────────────────────────

@dataclass
class OCRConfig:
    # Язык OCR — русский + английский (цифры, латиница на штрихкодах)
    lang: str = "ru"

    # Использовать GPU (False = CPU, достаточно для нашей задачи)
    use_gpu: bool = False

    # Классификатор угла текста — важно для повёрнутых ценников
    use_angle_cls: bool = True

    # Минимальная уверенность OCR (0–1), ниже — игнорируем блок
    min_confidence: float = 0.5

    # Попытки чтения штрихкода: исходный + инвертированный + увеличенный
    barcode_attempts: int = 3


# ─────────────────────────────────────────────
# Структура результата
# ─────────────────────────────────────────────

# Все поля которые требует задание
EMPTY_RESULT = {
    # Данные с ценника
    "product_name":    "",
    "price_default":   "",
    "price_card":      "",
    "price_discount":  "",
    "barcode":         "",
    "discount_amount": "",
    "id_sku":          "",
    "print_datetime":  "",
    "code":            "",
    "additional_info": "",
    "color":           "",
    "special_symbols": "",

    # Данные из QR
    "qr_code_barcode":        "",
    "price1_qr":              "",
    "price2_qr":              "",
    "price3_qr":              "",
    "price4_qr":              "",
    "wholesale_level_1_count":"",
    "wholesale_level_1_price":"",
    "wholesale_level_2_count":"",
    "wholesale_level_2_price":"",
    "action_price_qr":        "",
    "action_code_qr":         "",
}

# Поля которых нет на конкретном ценнике → "нет"
# Поля которые есть но не распознаны → "" (пусто)


# ─────────────────────────────────────────────
# Regex паттерны для парсинга текста
# ─────────────────────────────────────────────

# Цена: число с опциональной дробной частью
# Примеры: 1104, 1104.99, 1104,99
# Исключаем числа с более чем 2 цифрами после точки (не цена)
RE_PRICE = re.compile(r'(?<!\d)(\d{2,5}(?:[.,]\d{2})?)(?!\d)')

# Скидка в процентах: -25%, 25%
RE_DISCOUNT_PCT = re.compile(r'-?(\d{1,3})\s*%')

# Скидка в рублях: -500р, 500 руб
RE_DISCOUNT_RUB = re.compile(r'-?(\d{2,5})\s*(?:р|руб|₽)')

# Штрихкод: 8, 12 или 13 цифр подряд
RE_BARCODE = re.compile(r'\b(\d{8}|\d{12}|\d{13})\b')

# Артикул (id_sku): обычно 4-8 цифр после слова "арт" или в нижней части
RE_SKU = re.compile(r'(?:арт\.?\s*|sku\s*)(\d{4,8})', re.IGNORECASE)

# Дата печати: ДД.ММ.ГГГГ ЧЧ:ММ или похожие форматы
RE_DATETIME = re.compile(
    r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)'
)

# Тип выкладки
RE_SPECIAL = re.compile(r'\b([шклШКЛ])\b')

# Код зоны выкладки: буква + цифры, например A1, B12
RE_CODE = re.compile(r'\b([A-ZА-Я]\d{1,3})\b')


# ─────────────────────────────────────────────
# QR-парсер
# ─────────────────────────────────────────────

def _parse_qr_data(raw: str) -> dict:
    """
    Парсит содержимое QR-кода ценника Ленты.
    Формат: ключ=значение разделённые & или ;
    Поддерживает как полные имена так и сокращения:
      barcode/b, price1/p1, price2/p2 ... price4/p4,
      wholesaleLevel1Count/wL1C, wholesaleLevel1Price/wL1P,
      wholesaleLevel2Count/wL2C, wholesaleLevel2Price/wL2P,
      actionPrice/aP, actionCode/aC
    """
    result = {k: "" for k in [
        "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
        "wholesale_level_1_count", "wholesale_level_1_price",
        "wholesale_level_2_count", "wholesale_level_2_price",
        "action_price_qr", "action_code_qr",
    ]}

    # Карта сокращений → полное имя поля
    KEY_MAP = {
        "barcode": "qr_code_barcode", "b":    "qr_code_barcode",
        "price1":  "price1_qr",       "p1":   "price1_qr",
        "price2":  "price2_qr",       "p2":   "price2_qr",
        "price3":  "price3_qr",       "p3":   "price3_qr",
        "price4":  "price4_qr",       "p4":   "price4_qr",
        "wholesalelevel1count": "wholesale_level_1_count",
        "wl1c":                 "wholesale_level_1_count",
        "wholesalelevel1price": "wholesale_level_1_price",
        "wl1p":                 "wholesale_level_1_price",
        "wholesalelevel2count": "wholesale_level_2_count",
        "wl2c":                 "wholesale_level_2_count",
        "wholesalelevel2price": "wholesale_level_2_price",
        "wl2p":                 "wholesale_level_2_price",
        "actionprice": "action_price_qr", "ap": "action_price_qr",
        "actioncode":  "action_code_qr",  "ac": "action_code_qr",
    }

    # Разбиваем по разделителям
    pairs = re.split(r'[&;|\n]', raw)
    for pair in pairs:
        if '=' not in pair:
            continue
        k, _, v = pair.partition('=')
        k = k.strip().lower().replace('-', '').replace('_', '')
        v = v.strip()
        mapped = KEY_MAP.get(k)
        if mapped:
            result[mapped] = v

    return result


# ─────────────────────────────────────────────
# Штрихкод / QR через pyzbar
# ─────────────────────────────────────────────

def _decode_codes(img: np.ndarray, attempts: int = 3) -> tuple[str, dict]:
    """
    Пробует декодировать штрихкоды и QR с изображения.
    Делает несколько попыток: исходный → увеличенный → инвертированный.

    Returns:
        barcode (str): строка штрихкода или ""
        qr_data (dict): распарсенные поля QR или пустой dict
    """
    try:
        from pyzbar import pyzbar
    except (ImportError, Exception) as e:
        logger.warning(f"pyzbar недоступен ({e}) — пробуем cv2 QR fallback")
        return _decode_codes_cv2(img)

    barcode_val = ""
    qr_data = {}

    variants = [img]

    if attempts >= 2:
        # Увеличенный вариант — помогает с мелкими штрихкодами
        h, w = img.shape[:2]
        big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(big)

    if attempts >= 3:
        # Инвертированный — некоторые QR лучше читаются инвертированными
        variants.append(cv2.bitwise_not(img))

    for variant in variants:
        gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)

        for obj in decoded:
            raw = obj.data.decode("utf-8", errors="ignore")

            if obj.type == "QRCODE":
                qr_data = _parse_qr_data(raw)
                logger.debug(f"QR decoded: {raw[:60]}...")

            elif obj.type in ("EAN13", "EAN8", "CODE128", "CODE39", "UPC_A", "UPC_E"):
                if not barcode_val:
                    barcode_val = raw
                    logger.debug(f"Barcode decoded: {raw}")

        # Если нашли и штрихкод и QR — хватит
        if barcode_val and qr_data:
            break

    return barcode_val, qr_data


# ─────────────────────────────────────────────
# EDSR Super Resolution для QR
# ─────────────────────────────────────────────

_sr_model = None  # кешируем модель между вызовами

def _get_sr_model(model_path: str = "models/EDSR_x4.pb"):
    """Загружает EDSR модель один раз и кеширует."""
    global _sr_model
    if _sr_model is not None:
        return _sr_model

    from pathlib import Path
    if not Path(model_path).exists():
        logger.warning(f"EDSR модель не найдена: {model_path}. QR будет читаться без SR.")
        return None

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", 4)
        _sr_model = sr
        logger.info("EDSR x4 загружен успешно")
        return _sr_model
    except Exception as e:
        logger.warning(f"Не удалось загрузить EDSR: {e}")
        return None


def _enhance_qr_zone(img: np.ndarray, model_path: str = "models/EDSR_x4.pb") -> np.ndarray:
    """
    Улучшает зону QR-кода:
    1. Вырезаем правый нижний угол (там QR на ценниках Ленты)
    2. Применяем EDSR x4 если модель доступна
    3. Морфологическая коррекция для чёткости модулей QR
    """
    h, w = img.shape[:2]

    # Вырезаем зону QR — правый нижний угол белой части ценника
    # QR на ценниках Ленты находится в правом нижнем углу
    qr_zone = img[int(h * 0.4):h, int(w * 0.35):w]

    if qr_zone.size == 0:
        return img

    # EDSR апскейл если модель доступна
    sr = _get_sr_model(model_path)
    if sr is not None:
        try:
            qr_upscaled = sr.upsample(qr_zone)
        except Exception as e:
            logger.debug(f"EDSR ошибка: {e}, используем INTER_LANCZOS4")
            qh, qw = qr_zone.shape[:2]
            qr_upscaled = cv2.resize(qr_zone, (qw * 4, qh * 4), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Fallback: обычный апскейл
        qh, qw = qr_zone.shape[:2]
        qr_upscaled = cv2.resize(qr_zone, (qw * 4, qh * 4), interpolation=cv2.INTER_LANCZOS4)

    # Возвращаем цветное изображение — cv2 QRCodeDetector лучше работает с BGR
    # Морфология убрана — она деформирует модули QR
    return qr_upscaled


def _decode_codes_cv2(img: np.ndarray) -> tuple[str, dict]:
    """
    Декодер QR через cv2 с EDSR улучшением.
    Читает только QR, не читает EAN/Code128 штрихкоды.
    """
    qr_detector = cv2.QRCodeDetector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qr_data = {}

    # Сначала пробуем улучшенную зону QR через EDSR
    enhanced = _enhance_qr_zone(img)

    # Готовим варианты для детектора — включая повёрнутые версии
    # QR на ценниках повёрнут на 90° вместе с ценником
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced

    variants = [
        ("enhanced",          enhanced_gray),
        ("enhanced_rot90cw",  cv2.rotate(enhanced_gray, cv2.ROTATE_90_CLOCKWISE)),
        ("enhanced_rot90ccw", cv2.rotate(enhanced_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("enhanced_rot180",   cv2.rotate(enhanced_gray, cv2.ROTATE_180)),
        ("original",          gray),
        ("orig_rot90cw",      cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        ("orig_rot90ccw",     cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("inverted_enhanced", cv2.bitwise_not(enhanced_gray)),
    ]

    for label, variant in variants:
        if variant is None:
            continue
        try:
            data, points, _ = qr_detector.detectAndDecode(variant)
            if data:
                logger.debug(f"cv2 QR decoded ({label}): {data[:60]}")
                qr_data = _parse_qr_data(data)
                return "", qr_data
        except Exception:
            continue

    return "", qr_data


# ─────────────────────────────────────────────
# Текстовый парсер
# ─────────────────────────────────────────────

def _parse_text_blocks(blocks: list[dict]) -> dict:
    """
    Принимает список OCR-блоков вида:
        {"text": "...", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}
    Возвращает словарь с распознанными полями ценника.

    Стратегия:
    - Сортируем блоки по размеру шрифта (высоте bbox) — цена самая крупная
    - Ищем цену как самое крупное число
    - Скидку — блок с % рядом с кружком
    - Дату, артикул, тип выкладки — по regex
    - Название товара — самый длинный текстовый блок без цифр
    """
    result = {}

    if not blocks:
        return result

    # Сортируем по высоте bbox (крупный текст = важный)
    blocks_sorted = sorted(
        blocks,
        key=lambda b: (b["bbox"][3] - b["bbox"][1]),
        reverse=True
    )

    all_text = " ".join(b["text"] for b in blocks)
    prices_found = []
    texts_found  = []

    for b in blocks:
        txt  = b["text"].strip()
        h    = b["bbox"][3] - b["bbox"][1]  # высота блока

        if not txt:
            continue

        # ── Скидка % ──────────────────────────────────────────
        # Блок вида '36%1104' — извлекаем скидку И цену отдельно
        m = RE_DISCOUNT_PCT.search(txt)
        if m and "discount_amount" not in result:
            result["discount_amount"] = f"-{m.group(1)}%"
            # Из этого блока берём только число ПОСЛЕ % как цену
            # Число ДО % — это процент скидки, не цена
            after_pct = txt[m.end():]
            price_match = RE_PRICE.search(after_pct)
            if price_match:
                try:
                    val = float(price_match.group(1).replace(",", "."))
                    if 50 <= val <= 99999:
                        prices_found.append((val, h, txt))
                except ValueError:
                    pass
            continue  # не парсим весь блок через общий цикл

        # ── Скидка в рублях ───────────────────────────────────
        m = RE_DISCOUNT_RUB.search(txt)
        if m and "discount_amount" not in result:
            result["discount_amount"] = f"-{m.group(1)}₽"

        # ── Дата/время ────────────────────────────────────────
        m = RE_DATETIME.search(txt)
        if m and "print_datetime" not in result:
            result["print_datetime"] = m.group(1)

        # ── Артикул ───────────────────────────────────────────
        m = RE_SKU.search(txt)
        if m and "id_sku" not in result:
            result["id_sku"] = m.group(1)

        # ── Тип выкладки ──────────────────────────────────────
        m = RE_SPECIAL.search(txt)
        if m and "special_symbols" not in result:
            result["special_symbols"] = m.group(1).lower()

        # ── Код зоны ──────────────────────────────────────────
        m = RE_CODE.search(txt)
        if m and "code" not in result:
            result["code"] = m.group(1)

        # ── Штрихкод из текста (fallback если pyzbar не сработал) ──
        m = RE_BARCODE.search(txt)
        if m and "barcode" not in result:
            result["barcode"] = m.group(1)

        # ── Цена ──────────────────────────────────────────────
        prices = RE_PRICE.findall(txt)
        for p in prices:
            try:
                val = float(p.replace(",", "."))
                if not (50 <= val <= 99999):
                    continue
                # Артефакт: 17472 → 1747 (лишняя '2' = символ рубля)
                p_int = p.replace(",", ".").split(".")[0]
                if len(p_int) == 5 and p_int.endswith("2"):
                    val = float(p_int[:-1])
                prices_found.append((val, h, txt))
            except ValueError:
                pass

        # ── Текст для названия товара ─────────────────────────
        # Собираем блоки где букв больше чем цифр
        # Сохраняем с позицией Y для сортировки по порядку чтения
        letters = sum(1 for c in txt if c.isalpha())
        digits  = sum(1 for c in txt if c.isdigit())
        if letters > digits and len(txt) > 3:
            y_pos = b["bbox"][1]  # верхняя координата блока
            texts_found.append((y_pos, txt))

    # ── Присваиваем цены ──────────────────────────────────────
    # Сортируем по высоте блока (крупнее = основная цена)
    prices_found.sort(key=lambda x: x[1], reverse=True)

    # Дедупликация цен — убираем одинаковые значения
    seen_prices = []
    for val, h, txt in prices_found:
        if val not in seen_prices:
            seen_prices.append(val)

    if seen_prices:
        result["price_card"] = str(seen_prices[0])
        if len(seen_prices) >= 2:
            result["price_default"] = str(seen_prices[1])

    # ── Название товара ───────────────────────────────────────
    # Сортируем по Y-позиции (сверху вниз) и объединяем все текстовые блоки
    # Это даёт правильный порядок строк названия товара
    if texts_found:
        texts_found.sort(key=lambda x: x[0])  # сортировка по Y
        # Объединяем все блоки через пробел
        combined = " ".join(t for _, t in texts_found)
        # Убираем дубли слов (одно слово из разных блоков)
        words = combined.split()
        seen_words = []
        for w in words:
            if w.lower() not in [s.lower() for s in seen_words]:
                seen_words.append(w)
        result["product_name"] = " ".join(seen_words)

    return result


# ─────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────

class OCREngine:
    """
    Читает текст с кропа ценника и возвращает структурированный словарь.

    Пример:
        engine = OCREngine(config)
        result = engine.read(crop, color="orange")
        print(result["price_card"])   # "1104"
        print(result["barcode"])      # "4607039374155"
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._ocr = None
        self._load_ocr()

    def _load_ocr(self):
        """Инициализирует PaddleOCR. Первый запуск скачивает модели (~100 MB)."""
        try:
            # Подавляем DEBUG-логи PaddleOCR до импорта
            import logging as _logging
            for _name in ["ppocr", "paddle", "ppdet"]:
                _logging.getLogger(_name).setLevel(_logging.ERROR)

            # Также через переменные окружения
            import os
            os.environ["PPOCR_LOG_LEVEL"] = "ERROR"

            from paddleocr import PaddleOCR
            import inspect

            # Подавляем ещё раз после импорта (PaddleOCR может переинициализировать)
            for _name in ["ppocr", "paddle", "ppdet", "root"]:
                _l = _logging.getLogger(_name)
                _l.setLevel(_logging.ERROR)
                for _h in _l.handlers:
                    _h.setLevel(_logging.ERROR)

            sig_params = inspect.signature(PaddleOCR.__init__).parameters

            kwargs = {}
            if "lang" in sig_params:
                kwargs["lang"] = self.config.lang
            if "use_textline_orientation" in sig_params:
                kwargs["use_textline_orientation"] = True
            elif "use_angle_cls" in sig_params:
                kwargs["use_angle_cls"] = True  # принудительно — иначе повёрнутый текст не читается
            if "use_gpu" in sig_params:
                kwargs["use_gpu"] = self.config.use_gpu
            # show_log всегда False — убираем DEBUG спам
            if "show_log" in sig_params:
                kwargs["show_log"] = False
            else:
                # Для версий где show_log не в __init__ — патчим через атрибут
                import os
                os.environ["PPOCR_LOG_LEVEL"] = "WARNING"

            self._ocr = PaddleOCR(**kwargs)
            logger.info(f"PaddleOCR загружен (параметры: {list(kwargs.keys())})")
        except ImportError:
            logger.error(
                "paddleocr не установлен. "
                "Установи: pip install paddlepaddle paddleocr"
            )
            raise

    def read(
        self,
        crop: np.ndarray,
        color: str = "",
        filename: str = "",
        frame_timestamp: int = 0,
        x_min: int = 0, y_min: int = 0,
        x_max: int = 0, y_max: int = 0,
    ) -> dict:
        """
        Полный пайплайн чтения одного кропа ценника.

        Args:
            crop:            BGR изображение ценника
            color:           цвет ценника из детектора
            filename:        имя видеофайла (для CSV)
            frame_timestamp: время кадра в мс
            x_min/y_min/x_max/y_max: координаты в исходном кадре

        Returns:
            dict со всеми полями для CSV (незаполненные = "")
        """
        result = dict(EMPTY_RESULT)

        # Служебные поля
        result["filename"]        = filename
        result["color"]           = color
        result["frame_timestamp"] = frame_timestamp
        result["x_min"]           = x_min
        result["y_min"]           = y_min
        result["x_max"]           = x_max
        result["y_max"]           = y_max

        # ── Шаг 1: Штрихкоды и QR ────────────────────────────────────────
        barcode, qr_data = _decode_codes(crop, self.config.barcode_attempts)

        if barcode:
            result["barcode"] = barcode

        if qr_data:
            result.update({k: v for k, v in qr_data.items() if v})

        # ── Шаг 2: PaddleOCR ─────────────────────────────────────────────
        ocr_blocks = self._run_paddle(crop)

        # ── Шаг 3: Парсинг текстовых блоков ──────────────────────────────
        parsed = _parse_text_blocks(ocr_blocks)
        for k, v in parsed.items():
            if v and not result.get(k):
                result[k] = v

        # ── Шаг 4: Если штрихкод не нашёлся через pyzbar — берём из текста
        if not result["barcode"] and parsed.get("barcode"):
            result["barcode"] = parsed["barcode"]

        # ── Шаг 5: Проставляем "нет" для отсутствующих полей ─────────────
        # Поля которых заведомо нет на оранжевом ценнике
        self._mark_absent(result, color)

        return result

    def _run_paddle(self, img: np.ndarray) -> list[dict]:
        """
        Запускает PaddleOCR и возвращает список блоков:
        [{"text": "...", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}, ...]
        """
        try:
            raw = self._ocr.ocr(img, cls=True)  # cls=True критично для повёрнутых ценников
        except Exception as e:
            logger.warning(f"PaddleOCR ошибка: {e}")
            return []

        blocks = []
        if not raw or not raw[0]:
            return blocks

        for line in raw[0]:
            if not line:
                continue
            try:
                bbox_pts, (text, conf) = line
            except Exception:
                continue
            if conf < self.config.min_confidence:
                continue
            if not text or not text.strip():
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            blocks.append({
                "text":       text.strip(),
                "confidence": float(conf),
                "bbox":       [min(xs), min(ys), max(xs), max(ys)],
            })

        logger.debug(f"PaddleOCR: {len(blocks)} блоков текста")
        return blocks

    def _mark_absent(self, result: dict, color: str):
        """
        Проставляет 'нет' для полей которых заведомо нет на данном типе ценника.
        Согласно заданию: отсутствующее поле = 'нет', нераспознанное = ''.
        """
        # Оптовые пороги — только на специальных ценниках (не на обычных orange)
        if color in ("orange", "blue", "white"):
            for f in ["wholesale_level_1_count", "wholesale_level_1_price",
                      "wholesale_level_2_count", "wholesale_level_2_price"]:
                if not result[f]:
                    result[f] = "нет"

        # Акционная цена — только если нет скидки
        if not result["price_discount"] and not result["discount_amount"]:
            if not result["action_price_qr"]:
                result["action_price_qr"] = "нет"
            if not result["action_code_qr"]:
                result["action_code_qr"] = "нет"


# ─────────────────────────────────────────────
# CLI для отладки
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Использование: python ocr_engine.py <crop.jpg>")
        print("              python ocr_engine.py <папка_с_кропами/>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    try:
        from config import OCR_CONFIG as config
        print("Конфиг загружен из config.py")
    except (ImportError, AttributeError):
        config = OCRConfig()
        print("Используются дефолтные настройки OCR")

    engine = OCREngine(config)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    # Один файл
    if input_path.is_file():
        img = cv2.imread(str(input_path))
        color = input_path.stem.split("_")[-1]

        # Показываем сырые OCR блоки для отладки
        print(f"\n{'─'*50}")
        print("Сырые OCR блоки:")
        print(f"{'─'*50}")
        blocks = engine._run_paddle(img)
        for i, b in enumerate(blocks):
            h = b["bbox"][3] - b["bbox"][1]
            print(f"  [{i+1:2d}] conf={b['confidence']:.2f} h={int(h):3d}px  '{b['text']}'")

        result = engine.read(img, color=color, filename=input_path.name)

        print(f"\n{'─'*50}")
        print(f"Файл: {input_path.name}")
        print(f"{'─'*50}")
        for k, v in result.items():
            if v and v != "нет":
                print(f"  {k:30s}: {v}")
        print(f"{'─'*50}\n")

    # Папка
    elif input_path.is_dir():
        crops = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"\nОбрабатываем {len(crops)} кропов из {input_path}\n")

        all_results = []
        empty_count = 0
        for crop_path in crops:
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            color = crop_path.stem.split("_")[-1]
            result = engine.read(img, color=color, filename=crop_path.name)

            price = result.get("price_card") or result.get("price_default") or ""
            name  = result.get("product_name", "")[:30]
            bc    = result.get("barcode", "")

            # Пропускаем полностью пустые результаты
            has_data = any([price, name, bc,
                           result.get("discount_amount"),
                           result.get("qr_code_barcode")])

            if has_data:
                all_results.append(result)
                print(f"  {crop_path.name:35s} цена={price:8s}  "
                      f"штрихкод={bc:14s}  товар={name}")
            else:
                empty_count += 1

        print(f"\nПустых кропов пропущено: {empty_count}")

        # Сохраняем JSON для проверки
        out = input_path / "ocr_results.json"
        out.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\nРезультаты сохранены в {out}")
