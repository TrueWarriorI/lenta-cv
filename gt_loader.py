"""
gt_loader.py — Загрузка ценников по ground truth разметке
==========================================================
Вместо детектора использует готовые координаты bbox из CSV-файла
который поставляется вместе с видео.

Преимущества:
- Точные координаты — никаких пропусков и ложных срабатываний
- Знаем frame_timestamp — извлекаем именно нужный кадр
- Можем сравнить наш OCR с эталонными значениями

Использование:
    loader = GTLoader("csv/25_2-10.csv", "videos/25_2-10.mp4")
    for crop, gt_record in loader.crops():
        ocr_result = ocr_engine.read(crop, ...)
        score = loader.evaluate(ocr_result, gt_record)
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Структура GT записи
# ─────────────────────────────────────────────

@dataclass
class GTRecord:
    """Одна запись из ground truth CSV."""
    filename:        str
    product_name:    str
    price_default:   str
    price_card:      str
    price_discount:  str
    barcode:         str
    discount_amount: str
    id_sku:          str
    print_datetime:  str
    code:            str
    additional_info: str
    color:           str
    special_symbols: str
    frame_timestamp: int   # мс
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    # QR поля
    qr_code_barcode:        str = ""
    price1_qr:              str = ""
    price2_qr:              str = ""
    price3_qr:              str = ""
    price4_qr:              str = ""
    wholesale_level_1_count: str = ""
    wholesale_level_1_price: str = ""
    wholesale_level_2_count: str = ""
    wholesale_level_2_price: str = ""
    action_price_qr:        str = ""
    action_code_qr:         str = ""


def _parse_coord(val: str) -> int:
    """Парсит координату с запятой как разделителем ('2051,8' → 2051)."""
    if not val:
        return 0
    # Заменяем запятую на точку и берём целую часть
    try:
        return int(float(val.replace(",", ".")))
    except ValueError:
        return 0


def _parse_timestamp(val: str) -> int:
    """Парсит timestamp в мс."""
    try:
        return int(float(val.replace(",", ".")))
    except ValueError:
        return 0


# ─────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────

class GTLoader:
    """
    Загружает ценники используя ground truth координаты из CSV.

    Пример:
        loader = GTLoader("csv/25_2-10.csv", "videos/25_2-10.mp4")

        # Получить все кропы
        for crop, gt in loader.crops():
            result = ocr.read(crop, color=gt.color)
            score = GTLoader.evaluate(result, gt)
            print(f"Точность: {score:.1%}")

        # Оценить качество OCR
        loader.run_evaluation(ocr_engine, preprocessor)
    """

    def __init__(
        self,
        csv_path: str | Path,
        video_path: str | Path,
        padding: int = 15,
    ):
        self.csv_path   = Path(csv_path)
        self.video_path = Path(video_path)
        self.padding    = padding
        self.records    = self._load_csv()

        logger.info(
            f"GTLoader: загружено {len(self.records)} записей "
            f"из {self.csv_path.name}"
        )

    def _load_csv(self) -> list[GTRecord]:
        """Читает и парсит ground truth CSV."""
        records = []

        with open(self.csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rec = GTRecord(
                        filename        = row.get("filename", ""),
                        product_name    = row.get("product_name", ""),
                        price_default   = row.get("price_default", ""),
                        price_card      = row.get("price_card", ""),
                        price_discount  = row.get("price_discount", ""),
                        barcode         = row.get("barcode", ""),
                        discount_amount = row.get("discount_amount", ""),
                        id_sku          = row.get("id_sku", ""),
                        print_datetime  = row.get("print_datetime", ""),
                        code            = row.get("code", ""),
                        additional_info = row.get("additional_info", ""),
                        color           = row.get("color", ""),
                        special_symbols = row.get("special_symbols", ""),
                        frame_timestamp = _parse_timestamp(row.get("frame_timestamp", "0")),
                        x_min = _parse_coord(row.get("x_min", "0")),
                        y_min = _parse_coord(row.get("y_min", "0")),
                        x_max = _parse_coord(row.get("x_max", "0")),
                        y_max = _parse_coord(row.get("y_max", "0")),
                        qr_code_barcode         = row.get("qr_code_barcode", ""),
                        price1_qr               = row.get("price1_qr", ""),
                        price2_qr               = row.get("price2_qr", ""),
                        price3_qr               = row.get("price3_qr", ""),
                        price4_qr               = row.get("price4_qr", ""),
                        wholesale_level_1_count = row.get("wholesale_level_1_count", ""),
                        wholesale_level_1_price = row.get("wholesale_level_1_price", ""),
                        wholesale_level_2_count = row.get("wholesale_level_2_count", ""),
                        wholesale_level_2_price = row.get("wholesale_level_2_price", ""),
                        action_price_qr         = row.get("action_price_qr", ""),
                        action_code_qr          = row.get("action_code_qr", ""),
                    )
                    records.append(rec)
                except Exception as e:
                    logger.warning(f"Ошибка парсинга строки: {e}")

        return records

    def crops(self) -> Iterator[tuple[np.ndarray, GTRecord]]:
        """
        Генератор — извлекает кроп ценника из видео по GT координатам.
        Yields: (crop_bgr, gt_record)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        try:
            for rec in self.records:
                # Переходим к нужному кадру
                cap.set(cv2.CAP_PROP_POS_MSEC, rec.frame_timestamp)
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Fallback: берём первый кадр
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"Не удалось прочитать кадр t={rec.frame_timestamp}мс")
                    continue

                # Вырезаем bbox с padding
                x1 = max(0, rec.x_min - self.padding)
                y1 = max(0, rec.y_min - self.padding)
                x2 = min(frame_w, rec.x_max + self.padding)
                y2 = min(frame_h, rec.y_max + self.padding * 3)  # больше снизу

                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Невалидный bbox для записи: {rec.product_name}")
                    continue

                crop = frame[y1:y2, x1:x2].copy()

                if crop.size == 0:
                    continue

                yield crop, rec

        finally:
            cap.release()

    @staticmethod
    def evaluate(ocr_result: dict, gt: GTRecord) -> float:
        """
        Оценивает точность OCR для одного ценника.
        Считает долю правильно распознанных полей из ключевых.

        Метрика задания: считаем ценник успешным если точность >= 80%.
        """
        # Ключевые поля для оценки (согласно заданию)
        KEY_FIELDS = [
            "price_card",
            "price_default",
            "barcode",
            "discount_amount",
            "product_name",
        ]

        correct = 0
        total   = 0

        for field in KEY_FIELDS:
            gt_val  = getattr(gt, field, "").strip()
            ocr_val = str(ocr_result.get(field, "")).strip()

            # Пропускаем поля которых нет на ценнике
            if gt_val in ("", "нет"):
                continue

            total += 1

            # Нормализуем для сравнения
            gt_norm  = _normalize_value(gt_val, field)
            ocr_norm = _normalize_value(ocr_val, field)

            if gt_norm and ocr_norm and gt_norm == ocr_norm:
                correct += 1
            elif gt_norm and ocr_norm:
                # Частичное совпадение для длинных строк (название товара)
                if field == "product_name":
                    # Считаем долю совпадающих слов
                    gt_words  = set(gt_norm.lower().split())
                    ocr_words = set(ocr_norm.lower().split())
                    if gt_words:
                        overlap = len(gt_words & ocr_words) / len(gt_words)
                        if overlap >= 0.5:  # 50% слов совпадают
                            correct += 0.5

        if total == 0:
            return 1.0  # нечего оценивать

        return correct / total

    def run_evaluation(self, ocr_engine, preprocessor) -> dict:
        """
        Запускает полный цикл оценки: GT координаты → кроп → OCR → метрика.
        Возвращает итоговую статистику.
        """
        results = []
        successful = 0  # ценники с точностью >= 80%

        for i, (crop, gt) in enumerate(self.crops()):
            # Предобработка
            processed = preprocessor.process_raw(crop, gt.color)
            if processed is None:
                processed = crop

            # OCR
            ocr_result = ocr_engine.read(
                processed,
                color=gt.color,
                filename=gt.filename,
                frame_timestamp=gt.frame_timestamp,
                x_min=gt.x_min, y_min=gt.y_min,
                x_max=gt.x_max, y_max=gt.y_max,
            )

            # Оценка
            score = self.evaluate(ocr_result, gt)

            results.append({
                "index":       i + 1,
                "product":     gt.product_name[:40],
                "gt_price":    gt.price_card,
                "ocr_price":   ocr_result.get("price_card", ""),
                "score":       score,
                "success":     score >= 0.8,
            })

            if score >= 0.8:
                successful += 1

            logger.info(
                f"[{i+1:2d}/{len(self.records)}] "
                f"score={score:.0%} "
                f"{'✅' if score >= 0.8 else '❌'} "
                f"{gt.product_name[:30]}"
            )

        total = len(results)
        success_rate = successful / total if total > 0 else 0

        summary = {
            "total":        total,
            "successful":   successful,
            "success_rate": success_rate,
            "results":      results,
        }

        print(f"\n{'='*50}")
        print(f"ИТОГ ОЦЕНКИ")
        print(f"{'='*50}")
        print(f"Всего ценников:     {total}")
        print(f"Успешных (≥80%):    {successful}")
        print(f"Итоговая метрика:   {success_rate:.1%}")
        print(f"{'='*50}\n")

        return summary


def _normalize_value(val: str, field: str) -> str:
    """Нормализует значение для сравнения."""
    if not val:
        return ""

    val = val.strip()

    if field in ("price_card", "price_default", "price_discount"):
        # Цены: сравниваем только целую часть (OCR не читает копейки)
        import re
        nums = re.findall(r'\d+[.,]?\d*', val)
        if nums:
            return str(int(float(nums[0].replace(",", "."))))
        return ""

    if field == "barcode":
        # Штрихкод: только цифры
        import re
        return re.sub(r'\D', '', val)

    if field == "discount_amount":
        # Скидка: нормализуем '-36%' и '36%' как одно
        import re
        m = re.search(r'(\d+)', val)
        return m.group(1) if m else ""

    # Для текстовых полей — нижний регистр
    return val.lower().strip()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 3:
        print("Использование: python gt_loader.py <gt.csv> <video.mp4>")
        print("  Пример: python gt_loader.py csv/25_2-10.csv videos/25_2-10.mp4")
        sys.exit(1)

    csv_path   = sys.argv[1]
    video_path = sys.argv[2]

    from preprocessor import Preprocessor, PreprocessorConfig
    from ocr_engine   import OCREngine

    try:
        from config import PREPROCESSOR_CONFIG, OCR_CONFIG
    except ImportError:
        PREPROCESSOR_CONFIG = PreprocessorConfig()
        OCR_CONFIG = None

    # Добавляем метод process_raw в Preprocessor
    class PreprocessorWithRaw(Preprocessor):
        def process_raw(self, crop, color=""):
            """Обработка уже вырезанного кропа."""
            import cv2
            cfg = self.config
            ch, cw = crop.shape[:2]

            # Поворот если вертикальный
            if ch > cw * 1.3:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ch, cw = cw, ch

            # Апскейл до 600px
            TARGET_WIDTH = 600
            if cw < TARGET_WIDTH:
                scale = TARGET_WIDTH / cw
                crop = cv2.resize(
                    crop,
                    (int(cw * scale), int(ch * scale)),
                    interpolation=cv2.INTER_LANCZOS4
                )

            return crop

    preprocessor = PreprocessorWithRaw(PREPROCESSOR_CONFIG)
    ocr_engine   = OCREngine(OCR_CONFIG)

    loader = GTLoader(csv_path, video_path)

    print(f"Загружено {len(loader.records)} GT записей")
    print(f"Запускаем оценку...\n")

    summary = loader.run_evaluation(ocr_engine, preprocessor)

    # Сохраняем детальные результаты
    import json
    out = Path("output/gt_evaluation.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Детальные результаты: {out}")
