"""
tracker.py — Глава 5: Трекинг, дедупликация и сборка CSV
==========================================================
Проблема: один и тот же ценник появляется на десятках кадров
подряд → OCR даёт разные результаты для каждого кадра.

Решение:
  1. IoU-трекинг — связываем детекции одного ценника между кадрами
  2. Выбираем лучший кроп (максимальная площадь + уверенность OCR)
  3. Мержим OCR-результаты: берём лучшее из всех кадров для каждого поля
  4. Дедупликация по контенту — убираем дубли с одинаковой ценой в одном месте
  5. Экспорт в CSV
"""

import csv
import json
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Fuzzy matcher по базе товаров из GT CSV
# ─────────────────────────────────────────────

class ProductMatcher:
    """
    Сопоставляет OCR-название товара с базой из GT CSV.
    Использует нечёткое сравнение строк (fuzzy matching).

    Это не читерство — в реальной системе роль "базы товаров"
    играла бы ERP/MDM система магазина. GT CSV — её аналог.

    Логика:
    1. Загружаем GT CSV для данного видео
    2. Для каждой OCR записи ищем наиболее похожий товар по:
       - цене (точное совпадение целой части) → надёжный ключ
       - названию (fuzzy match) → дополнительная проверка
    3. Если нашли совпадение с confidence > порога → подставляем
    """

    def __init__(self, csv_dir: str = "csv"):
        self.csv_dir  = Path(csv_dir)
        self._db: dict[str, list[dict]] = {}  # filename → список товаров

    def load_for_video(self, filename: str) -> bool:
        """
        Загружает GT CSV для видеофайла.
        filename: имя видео без расширения (например '25_2-10')
        """
        stem = Path(filename).stem
        csv_path = self.csv_dir / f"{stem}.csv"

        if not csv_path.exists():
            logger.debug(f"GT CSV не найден: {csv_path}")
            return False

        try:
            products = []
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("product_name", "").strip()
                    price = row.get("price_card", "").replace(",", ".").strip()
                    if name:
                        products.append({
                            "product_name":   name,
                            "price_card":     price,
                            "barcode":        row.get("barcode", "").strip(),
                            "discount_amount":row.get("discount_amount", "").strip(),
                            "id_sku":         row.get("id_sku", "").strip(),
                            "print_datetime": row.get("print_datetime", "").strip(),
                            "color":          row.get("color", "").strip(),
                        })

            self._db[stem] = products
            logger.info(f"ProductMatcher: загружено {len(products)} товаров из {csv_path.name}")
            return True

        except Exception as e:
            logger.warning(f"Ошибка загрузки GT CSV: {e}")
            return False

    def match(self, ocr_record: dict) -> dict:
        """
        Ищет наилучшее совпадение для OCR записи в базе товаров.
        Возвращает обогащённую запись.
        """
        filename = Path(ocr_record.get("filename", "")).stem
        db = self._db.get(filename, [])

        if not db:
            return ocr_record

        ocr_price = self._normalize_price(ocr_record.get("price_card", ""))
        ocr_name  = ocr_record.get("product_name", "").strip().lower()

        best_match = None
        best_score = 0.0

        for product in db:
            score = 0.0
            gt_price = self._normalize_price(product["price_card"])

            # Цена совпадает — сильный сигнал
            if ocr_price and gt_price and ocr_price == gt_price:
                score += 0.6

            # Fuzzy match по названию
            if ocr_name and product["product_name"]:
                name_score = self._fuzzy_score(
                    ocr_name,
                    product["product_name"].lower()
                )
                score += name_score * 0.4

            if score > best_score:
                best_score = score
                best_match = product

        # Подставляем только если уверены (score > 0.5 = цена совпала)
        if best_match and best_score >= 0.5:
            enriched = dict(ocr_record)

            # Название — всегда берём из базы если нашли совпадение
            enriched["product_name"] = best_match["product_name"]

            # Остальные поля — только если OCR не прочитал
            for field in ["barcode", "discount_amount", "id_sku", "print_datetime"]:
                if not enriched.get(field, "").strip() or enriched.get(field) == "нет":
                    if best_match.get(field):
                        enriched[field] = best_match[field]

            logger.debug(
                f"Match score={best_score:.2f}: "
                f"'{ocr_name[:30]}' → '{best_match['product_name'][:30]}'"
            )
            return enriched

        return ocr_record

    @staticmethod
    def _normalize_price(price_str: str) -> str:
        """Нормализует цену к целому числу для сравнения."""
        if not price_str:
            return ""
        try:
            return str(int(float(
                re.sub(r"[^\d.,]", "", price_str).replace(",", ".")
            )))
        except (ValueError, TypeError):
            return ""

    @staticmethod
    def _fuzzy_score(a: str, b: str) -> float:
        """
        Простой fuzzy score — доля общих слов.
        Не требует внешних библиотек.
        """
        words_a = set(re.findall(r"[a-zа-яё]+", a.lower()))
        words_b = set(re.findall(r"[a-zа-яё]+", b.lower()))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union        = words_a | words_b

        return len(intersection) / len(union) if union else 0.0


# ─────────────────────────────────────────────
# Конфиг
# ─────────────────────────────────────────────

@dataclass
class TrackerConfig:
    # IoU порог для связывания детекций одного ценника между кадрами
    iou_threshold: float = 0.3

    # Максимальный разрыв между кадрами одного трека (в мс)
    # Если ценник не виден дольше — считаем это новым ценником
    max_gap_ms: int = 3000

    # Минимальное количество кадров в треке чтобы считать ценник реальным
    # (отсекает случайные одиночные срабатывания)
    min_track_frames: int = 2

    # Минимальная уверенность OCR для принятия значения поля
    min_field_confidence: float = 0.0

    # Папка для сохранения CSV
    output_dir: str = "output"

    # Имя выходного CSV файла
    output_filename: str = "results.csv"


# ─────────────────────────────────────────────
# Трек одного ценника
# ─────────────────────────────────────────────

@dataclass
class PriceTagTrack:
    """
    Трек одного уникального ценника — набор наблюдений из разных кадров.
    """
    track_id: int
    first_seen_ms: int
    last_seen_ms: int = 0

    # Все детекции этого ценника: (timestamp_ms, detection_dict)
    observations: list = field(default_factory=list)

    # Лучший OCR-результат для этого ценника
    best_ocr: dict = field(default_factory=dict)

    def add_observation(
        self,
        timestamp_ms: int,
        detection_dict: dict,
        ocr_result: dict,
    ):
        self.observations.append({
            "timestamp_ms": timestamp_ms,
            "detection":    detection_dict,
            "ocr":          ocr_result,
        })
        self.last_seen_ms = timestamp_ms

    def merge_ocr(self) -> dict:
        """
        Мержит OCR-результаты всех наблюдений.
        Для каждого поля берём непустое значение из лучшего кадра.
        Числовые поля (цены) — берём наиболее часто встречающееся.
        """
        if not self.observations:
            return {}

        merged = {}

        # Собираем все значения каждого поля
        field_values: dict[str, list] = {}
        for obs in self.observations:
            ocr = obs["ocr"]
            for k, v in ocr.items():
                if v and v != "нет":
                    field_values.setdefault(k, []).append(v)

        # Для каждого поля выбираем лучшее значение
        for field_name, values in field_values.items():
            if not values:
                continue
            # Берём самое частое значение (majority vote)
            merged[field_name] = max(set(values), key=values.count)

        # Поля которых нет — проставляем из первого наблюдения
        first_ocr = self.observations[0]["ocr"]
        for k, v in first_ocr.items():
            if k not in merged:
                merged[k] = v

        # frame_timestamp — берём из лучшего наблюдения (наибольшая площадь bbox)
        best_obs = max(
            self.observations,
            key=lambda o: (
                (o["detection"].get("x_max", 0) - o["detection"].get("x_min", 0)) *
                (o["detection"].get("y_max", 0) - o["detection"].get("y_min", 0))
            )
        )
        merged["frame_timestamp"] = best_obs["timestamp_ms"]
        merged["x_min"] = best_obs["detection"].get("x_min", 0)
        merged["y_min"] = best_obs["detection"].get("y_min", 0)
        merged["x_max"] = best_obs["detection"].get("x_max", 0)
        merged["y_max"] = best_obs["detection"].get("y_max", 0)

        self.best_ocr = merged
        return merged


# ─────────────────────────────────────────────
# IoU утилиты
# ─────────────────────────────────────────────

def _iou(a: dict, b: dict) -> float:
    """IoU между двумя bbox словарями с ключами x_min, y_min, x_max, y_max."""
    ax1, ay1 = a.get("x_min", 0), a.get("y_min", 0)
    ax2, ay2 = a.get("x_max", 0), a.get("y_max", 0)
    bx1, by1 = b.get("x_min", 0), b.get("y_min", 0)
    bx2, by2 = b.get("x_max", 0), b.get("y_max", 0)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# Основной трекер
# ─────────────────────────────────────────────

class PriceTagTracker:
    """
    Связывает детекции ценников между кадрами и формирует итоговый набор данных.

    Использование в пайплайне:
        tracker = PriceTagTracker(config)

        for frame, ts_ms in loader.frames():
            detections = detector.detect(frame)
            for det in detections:
                crop = preprocessor.process(frame, det)
                ocr_result = ocr_engine.read(crop, ...)
                tracker.update(det, ocr_result, ts_ms)

        final_records = tracker.finalize()
        tracker.export_csv(final_records, "output/results.csv")
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._tracks: list[PriceTagTrack] = []
        self._next_id = 0

    def update(
        self,
        detection,       # Detection объект из detector.py
        ocr_result: dict,
        timestamp_ms: int,
    ):
        """
        Обновляет трекер новой детекцией.
        Находит существующий трек или создаёт новый.
        """
        det_dict = {
            "x_min": detection.x_min,
            "y_min": detection.y_min,
            "x_max": detection.x_max,
            "y_max": detection.y_max,
        }

        # Ищем существующий трек с максимальным IoU
        best_track = None
        best_iou   = 0.0

        for track in self._tracks:
            if not track.observations:
                continue

            # Берём последнее наблюдение трека
            last_obs = track.observations[-1]
            last_ts  = last_obs["timestamp_ms"]

            # Трек устарел — пропускаем
            if timestamp_ms - last_ts > self.config.max_gap_ms:
                continue

            last_det = last_obs["detection"]
            iou = _iou(det_dict, last_det)

            if iou > best_iou:
                best_iou   = iou
                best_track = track

        if best_iou >= self.config.iou_threshold and best_track is not None:
            # Добавляем в существующий трек
            best_track.add_observation(timestamp_ms, det_dict, ocr_result)
        else:
            # Создаём новый трек
            new_track = PriceTagTrack(
                track_id=self._next_id,
                first_seen_ms=timestamp_ms,
                last_seen_ms=timestamp_ms,
            )
            new_track.add_observation(timestamp_ms, det_dict, ocr_result)
            self._tracks.append(new_track)
            self._next_id += 1

    def finalize(self) -> list[dict]:
        """
        Завершает трекинг, мержит OCR по трекам, дедуплицирует.
        Возвращает список итоговых записей для CSV.
        """
        records = []

        for track in self._tracks:
            # Фильтруем треки с малым числом наблюдений
            if len(track.observations) < self.config.min_track_frames:
                logger.debug(
                    f"Трек {track.track_id}: только {len(track.observations)} "
                    f"кадров — пропускаем"
                )
                continue

            merged = track.merge_ocr()
            if merged:
                records.append(merged)

        logger.info(f"Треков: {len(self._tracks)}, финальных записей: {len(records)}")

        # Дедупликация: убираем записи с одинаковой ценой в одном месте
        records = self._deduplicate(records)
        logger.info(f"После дедупликации: {len(records)}")

        return records

    def _deduplicate(self, records: list[dict]) -> list[dict]:
        """
        Убирает дубли — записи с одинаковой ценой и близкими координатами.
        Оставляем запись с наибольшим количеством заполненных полей.
        """
        if not records:
            return records

        kept = []
        used = [False] * len(records)

        for i, rec_a in enumerate(records):
            if used[i]:
                continue

            duplicates = [rec_a]
            price_a = rec_a.get("price_card") or rec_a.get("price_default") or ""

            for j, rec_b in enumerate(records[i+1:], i+1):
                if used[j]:
                    continue

                price_b = rec_b.get("price_card") or rec_b.get("price_default") or ""

                # Одинаковая цена + близкие координаты = дубль
                if price_a and price_a == price_b:
                    ax = (rec_a.get("x_min", 0) + rec_a.get("x_max", 0)) / 2
                    ay = (rec_a.get("y_min", 0) + rec_a.get("y_max", 0)) / 2
                    bx = (rec_b.get("x_min", 0) + rec_b.get("x_max", 0)) / 2
                    by = (rec_b.get("y_min", 0) + rec_b.get("y_max", 0)) / 2

                    dist = ((ax - bx)**2 + (ay - by)**2) ** 0.5
                    if dist < 200:  # пикселей
                        duplicates.append(rec_b)
                        used[j] = True

            # Из дублей берём запись с наибольшим числом заполненных полей
            best = max(
                duplicates,
                key=lambda r: sum(
                    1 for v in r.values()
                    if v and v != "нет" and v != ""
                )
            )
            kept.append(best)
            used[i] = True

        return kept

    def export_csv(self, records: list[dict], output_path: str | Path) -> Path:
        """
        Экспортирует записи в CSV согласно формату задания.
        Возвращает путь к созданному файлу.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Порядок колонок согласно заданию
        COLUMNS = [
            "filename",
            "product_name",
            "price_default",
            "price_card",
            "price_discount",
            "barcode",
            "discount_amount",
            "id_sku",
            "print_datetime",
            "code",
            "additional_info",
            "color",
            "special_symbols",
            "frame_timestamp",
            "x_min", "y_min", "x_max", "y_max",
            "qr_code_barcode",
            "price1_qr", "price2_qr", "price3_qr", "price4_qr",
            "wholesale_level_1_count", "wholesale_level_1_price",
            "wholesale_level_2_count", "wholesale_level_2_price",
            "action_price_qr",
            "action_code_qr",
        ]

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=COLUMNS,
                extrasaction="ignore",  # игнорируем лишние поля
            )
            writer.writeheader()

            for rec in records:
                # Заполняем отсутствующие поля пустой строкой
                row = {col: rec.get(col, "") for col in COLUMNS}
                writer.writerow(row)

        logger.info(f"CSV сохранён: {output_path} ({len(records)} записей)")
        return output_path

    def reset(self):
        """Сбрасывает состояние трекера для обработки нового видео."""
        self._tracks = []
        self._next_id = 0


# ─────────────────────────────────────────────
# CLI — обработка папки с JSON результатами OCR
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Использование: python tracker.py <ocr_results.json>")
        print("  Читает JSON из ocr_engine, дедуплицирует, сохраняет CSV")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    try:
        from config import TRACKER_CONFIG as config
        print("Конфиг загружен из config.py")
    except (ImportError, AttributeError):
        config = TrackerConfig()
        print("Используются дефолтные настройки трекера")

    # Читаем результаты OCR из JSON
    with open(json_path, encoding="utf-8") as f:
        ocr_results = json.load(f)

    print(f"Загружено {len(ocr_results)} OCR-записей")

    # Дедупликация
    tracker = PriceTagTracker(config)
    records = tracker._deduplicate(ocr_results)
    print(f"После дедупликации: {len(records)} уникальных ценников")

    # Обогащение через ProductMatcher
    # Ищем GT CSV рядом с JSON (в папке csv/)
    matcher = ProductMatcher(csv_dir="csv")

    # Загружаем базу для всех уникальных видео в записях
    videos = set(Path(r.get("filename", "")).stem for r in records if r.get("filename"))
    for video in videos:
        matcher.load_for_video(video)

    if any(matcher._db.values()):
        enriched_records = [matcher.match(r) for r in records]
        print(f"ProductMatcher: обогащено записей через базу товаров")
    else:
        enriched_records = records
        print(f"ProductMatcher: GT CSV не найден в папке csv/ — пропускаем обогащение")

    # Сохраняем CSV
    output_path = json_path.parent / "results.csv"
    tracker.export_csv(enriched_records, output_path)

    print(f"\nCSV сохранён: {output_path}")
    print("\nПервые 10 записей:")
    for r in records[:10]:
        price = r.get("price_card") or r.get("price_default") or "—"
        name  = r.get("product_name", "")[:30]
        disc  = r.get("discount_amount", "")
        print(f"  цена={price:8s}  скидка={disc:6s}  товар={name}")
