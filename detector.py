"""
detector.py — Глава 2: Детекция ценников
==========================================
Два метода, работающих вместе:

  1. YOLOv8 — основной детектор. Использует предобученную модель.
     Если специализированной модели нет — запускаем zero-shot через
     общую модель (ищем прямоугольные объекты у полок).

  2. HSV color mask — резервный метод (fallback).
     Ценники Ленты имеют характерные цвета: оранжевый/красный (акция),
     синий (РПЦ), белый, жёлтый, зелёный.
     Находим цветные прямоугольники нужного размера и формы.

Оба метода дают список Detection(bbox, confidence, method).
Результаты объединяются через NMS (non-maximum suppression).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────

@dataclass
class Detection:
    """Одно обнаруженное поле (ценник) на кадре."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    method: str          # "yolo" | "hsv"
    color: str = ""      # "orange" | "blue" | "white" | "yellow" | "green"

    @property
    def bbox(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def area(self):
        return self.width * self.height

    @property
    def aspect_ratio(self):
        return self.width / self.height if self.height > 0 else 0

    def to_dict(self):
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "color": self.color,
        }


@dataclass
class DetectorConfig:
    # ── YOLOv8 ──────────────────────────────
    # Путь к весам модели. Если None — используется yolov8n.pt (скачается сам)
    model_path: Optional[str] = None

    # Порог уверенности детектора
    confidence: float = 0.35

    # IoU порог для NMS внутри YOLO
    iou_threshold: float = 0.45

    # Классы YOLO которые считаем ценниками.
    # None = искать во всех классах (zero-shot режим через book/label классы)
    target_classes: Optional[list] = None

    # Использовать ли HSV-метод как дополнение к YOLO
    use_hsv_fallback: bool = True

    # Использовать ли только HSV (без YOLO) — для отладки
    hsv_only: bool = False

    # ── Фильтры размера ──────────────────────
    # Минимальный размер ценника в пикселях
    min_width: int = 30
    min_height: int = 20

    # Максимальный размер (отсекаем случайные большие прямоугольники)
    max_width: int = 800
    max_height: int = 600

    # Допустимое соотношение сторон (ценники обычно шире чем высокие)
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 8.0

    # ── NMS после объединения результатов ───
    final_iou_threshold: float = 0.4


# ─────────────────────────────────────────────
# HSV-цвета ценников Ленты
# ─────────────────────────────────────────────

# Каждый цвет: (name, lower_hsv, upper_hsv)
#
# ВАЖНО: белый и жёлтый НАМЕРЕННО исключены из HSV-детекции.
# Причина: на видео алкогольного отдела этикетки бутылок неотличимы
# от белых/жёлтых ценников по одному цвету — слишком много ложных срабатываний.
# Белые и жёлтые ценники должен находить YOLO.
# Если нужно вернуть — добавь цвет обратно в список и перепроверь качество.
PRICE_TAG_COLORS = [
    (
        # Оранжевый — главный акционный цвет Ленты.
        # S >= 130: достаточно чтобы отсечь бежевые/кремовые этикетки,
        # но не слишком высокий — ценники на краях fisheye бледнеют.
        # Hue 4..20 покрывает весь диапазон от красно-оранжевого до жёлто-оранжевого.
        "orange",
        np.array([4,  130, 130]),
        np.array([22, 255, 255]),
    ),
    (
        # Красный (нижняя часть Hue 0..4)
        "red",
        np.array([0,  130, 130]),
        np.array([4,  255, 255]),
    ),
    (
        # Красный wrap-around через 180°
        "red2",
        np.array([170, 130, 130]),
        np.array([180, 255, 255]),
    ),
    (
        # Синий РПЦ — специфичный цвет, конкурентов мало
        "blue",
        np.array([100, 90, 70]),
        np.array([125, 255, 255]),
    ),
    (
        # Зелёный (овощи-фрукты) — насыщенный, не путается с листьями
        "green",
        np.array([45, 110, 70]),
        np.array([75, 255, 200]),
    ),
    # "white"  — исключён: неотличим от этикеток бутылок
    # "yellow" — исключён: неотличим от этикеток бутылок
]

# Соответствие внутренних имён цветов → поле color в CSV
COLOR_CANONICAL = {
    "orange": "orange",
    "red":    "orange",   # оба считаем "orange/red" для CSV
    "red2":   "orange",
    "blue":   "blue",
    "yellow": "yellow",
    "green":  "green",
    "white":  "white",
}


# ─────────────────────────────────────────────
# HSV-детектор
# ─────────────────────────────────────────────

def _hsv_detect(
    frame: np.ndarray,
    config: DetectorConfig,
) -> list[Detection]:
    """
    Ищет ценники по характерным HSV-цветам.
    Возвращает список Detection с method="hsv".
    """
    frame_h, frame_w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, lower, upper in PRICE_TAG_COLORS:
        mask = cv2.inRange(hsv, lower, upper)

        # Для красного объединяем две маски (wrap-around по Hue)
        if color_name == "red":
            _, lower2, upper2 = PRICE_TAG_COLORS[2]  # red2
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        elif color_name == "red2":
            continue  # уже обработано в "red"

        # Морфология: убираем шум, заполняем дыры
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Размерные фильтры из config
            if w < config.min_width or h < config.min_height:
                continue
            if w > config.max_width or h > config.max_height:
                continue

            aspect = w / h if h > 0 else 0
            if not (config.min_aspect_ratio <= aspect <= config.max_aspect_ratio):
                continue

            # Насколько контур заполняет bbox (прямоугольность)
            rect_area = w * h
            cnt_area  = cv2.contourArea(cnt)
            fill_ratio = cnt_area / rect_area if rect_area > 0 else 0

            # Цветные ценники — прямоугольные объекты.
            # 0.50 = компромисс: ловим ценники под углом (fisheye даёт трапецию),
            # но отсекаем круглые крышки и неровные контуры.
            if fill_ratio < 0.50:
                continue

            # Синие крышки бутылок тоже синие — требуем минимальный размер.
            if color_name == "blue" and (w < config.min_width * 1.3 or h < config.min_height * 1.3):
                continue

            # Уверенность = fill_ratio (нет нейросети — оцениваем геометрией)
            confidence = min(0.85, 0.5 + fill_ratio * 0.4)

            # Расширяем bbox вправо для оранжевых/красных ценников.
            # Оранжевая полоса — левая часть ценника.
            # Справа от неё белая часть с названием товара, штрихкодом и QR.
            # Ширина белой части примерно равна 2-3x ширине оранжевой полосы.
            x_max = x + w
            if color_name in ("orange", "red", "red2"):
                expand = int(w * 2.5)
                x_max = min(frame_w, x + w + expand)

            detections.append(Detection(
                x_min=x, y_min=y,
                x_max=x_max, y_max=y + h,
                confidence=confidence,
                method="hsv",
                color=COLOR_CANONICAL.get(color_name, color_name),
            ))

    return detections


# ─────────────────────────────────────────────
# YOLO-детектор
# ─────────────────────────────────────────────

def _yolo_detect(
    frame: np.ndarray,
    model,
    config: DetectorConfig,
) -> list[Detection]:
    """
    Запускает YOLOv8 на кадре.
    Возвращает список Detection с method="yolo".
    """
    results = model(
        frame,
        conf=config.confidence,
        iou=config.iou_threshold,
        verbose=False,
    )

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])

            # Если заданы target_classes — фильтруем
            if config.target_classes and cls_id not in config.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            w = x2 - x1
            h = y2 - y1

            if w < config.min_width or h < config.min_height:
                continue
            if w > config.max_width or h > config.max_height:
                continue

            aspect = w / h if h > 0 else 0
            if not (config.min_aspect_ratio <= aspect <= config.max_aspect_ratio):
                continue

            detections.append(Detection(
                x_min=x1, y_min=y1,
                x_max=x2, y_max=y2,
                confidence=conf,
                method="yolo",
            ))

    return detections


# ─────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────

def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """
    Non-Maximum Suppression: убирает дублирующиеся bbox.
    Сортируем по confidence, жадно отбираем не перекрывающиеся.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []

    for det in detections:
        dominated = False
        for k in kept:
            if _iou(det, k) > iou_threshold:
                # Если YOLO и HSV нашли одно и то же — оставляем YOLO
                # (выше confidence), но копируем color из HSV если YOLO не знает
                if k.method == "yolo" and not k.color and det.color:
                    k.color = det.color
                dominated = True
                break
        if not dominated:
            kept.append(det)

    return kept


def _iou(a: Detection, b: Detection) -> float:
    """Intersection over Union двух bbox."""
    ix1 = max(a.x_min, b.x_min)
    iy1 = max(a.y_min, b.y_min)
    ix2 = min(a.x_max, b.x_max)
    iy2 = min(a.y_max, b.y_max)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────

class PriceTagDetector:
    """
    Детектор ценников. Объединяет YOLOv8 и HSV-метод.

    Пример:
        detector = PriceTagDetector(config)
        for frame, ts_ms in loader.frames():
            detections = detector.detect(frame)
            for d in detections:
                crop = frame[d.y_min:d.y_max, d.x_min:d.x_max]
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._model = None

        if not self.config.hsv_only:
            self._load_model()

    def _load_model(self):
        """Загружает YOLOv8. При первом запуске скачивает веса (~6 MB для nano)."""
        try:
            from ultralytics import YOLO

            model_path = self.config.model_path or "yolov8n.pt"
            logger.info(f"Загружаем YOLO модель: {model_path}")
            self._model = YOLO(model_path)
            logger.info("YOLO загружен успешно")

        except ImportError:
            logger.warning(
                "ultralytics не установлен — работаем только в HSV-режиме. "
                "Установи: pip install ultralytics"
            )
            self.config.use_hsv_fallback = True
            self.config.hsv_only = True

        except Exception as e:
            logger.warning(f"Не удалось загрузить YOLO ({e}) — переходим на HSV")
            self._model = None
            self.config.use_hsv_fallback = True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Запускает детекцию на одном кадре.
        Возвращает список Detection после NMS.
        """
        detections = []

        # YOLO
        if self._model is not None and not self.config.hsv_only:
            try:
                yolo_dets = _yolo_detect(frame, self._model, self.config)
                detections.extend(yolo_dets)
                logger.debug(f"YOLO: {len(yolo_dets)} детекций")
            except Exception as e:
                logger.warning(f"YOLO ошибка: {e}")

        # HSV fallback
        if self.config.use_hsv_fallback or self.config.hsv_only:
            hsv_dets = _hsv_detect(frame, self.config)
            detections.extend(hsv_dets)
            logger.debug(f"HSV: {len(hsv_dets)} детекций")

        # Объединяем через NMS
        result = _nms(detections, self.config.final_iou_threshold)
        logger.debug(f"После NMS: {len(result)} ценников")
        return result

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Рисует bbox на копии кадра для визуальной отладки.
        Возвращает аннотированный кадр (оригинал не изменяется).
        """
        COLOR_MAP = {
            "yolo": (0, 255, 0),    # зелёный — YOLO
            "hsv":  (0, 165, 255),  # оранжевый — HSV
        }
        vis = frame.copy()

        for d in detections:
            color = COLOR_MAP.get(d.method, (255, 255, 255))
            cv2.rectangle(vis, (d.x_min, d.y_min), (d.x_max, d.y_max), color, 2)

            label = f"{d.method} {d.confidence:.2f}"
            if d.color:
                label += f" [{d.color}]"

            # Фон под текст
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                vis,
                (d.x_min, d.y_min - th - 6),
                (d.x_min + tw + 4, d.y_min),
                color, -1
            )
            cv2.putText(
                vis, label,
                (d.x_min + 2, d.y_min - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA
            )

        return vis


# ─────────────────────────────────────────────
# CLI для отладки
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Использование: python detector.py <video_or_image>")
        print("  Пример (видео): python detector.py robot_01.mp4")
        print("  Пример (фото): python detector.py frame_00001000.jpg")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Конфиг
    try:
        from config import DETECTOR_CONFIG as config
        print("Конфиг загружен из config.py")
    except (ImportError, AttributeError):
        config = DetectorConfig(
            hsv_only=False,
            use_hsv_fallback=True,
            confidence=0.35,
        )
        print("Используются дефолтные настройки детектора")

    detector = PriceTagDetector(config)

    # ── Режим изображения ─────────────────────
    if input_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(str(input_path))
        if frame is None:
            print(f"Не удалось открыть: {input_path}")
            sys.exit(1)

        dets = detector.detect(frame)
        print(f"\nНайдено ценников: {len(dets)}")
        for i, d in enumerate(dets):
            print(f"  [{i+1}] {d.method:4s} conf={d.confidence:.2f} "
                  f"bbox=({d.x_min},{d.y_min},{d.x_max},{d.y_max}) "
                  f"size={d.width}x{d.height} color={d.color}")

        vis = detector.draw(frame, dets)
        out_path = input_path.parent / f"{input_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"\nРезультат сохранён: {out_path}")

    # ── Режим видео ───────────────────────────
    else:
        from video_loader import VideoLoader, VideoLoaderConfig

        try:
            from config import LOADER_CONFIG as loader_cfg
        except (ImportError, AttributeError):
            loader_cfg = VideoLoaderConfig(fps_sample=1.0, apply_clahe=True)

        output_dir = Path("detector_debug") / input_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        loader = VideoLoader(input_path, loader_cfg)
        total_found = 0
        frame_count = 0

        for frame, ts_ms in loader.frames():
            dets = detector.detect(frame)
            total_found += len(dets)
            frame_count += 1

            if dets:
                vis = detector.draw(frame, dets)
                out_path = output_dir / f"frame_{ts_ms:08d}_det{len(dets)}.jpg"
                cv2.imwrite(str(out_path), vis)

        print(f"\nОбработано кадров: {frame_count}")
        print(f"Всего детекций: {total_found}")
        print(f"Результаты в: {output_dir}/")
