"""
preprocessor.py — Глава 3: Коррекция перспективы и подготовка кропа
=====================================================================
Берёт кадр + Detection от детектора, возвращает изображение ценника
готовое для OCR:

  1. Вырезаем bbox с небольшим padding
  2. Выравниваем перспективу (ценники под углом из-за fisheye)
  3. Апскейл если ценник мелкий
  4. Финальная заточка под OCR: контраст, резкость, бинаризация (опц.)

Каждый шаг можно включать/выключать через PreprocessorConfig.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

from detector import Detection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Конфиг
# ─────────────────────────────────────────────

@dataclass
class PreprocessorConfig:
    # Отступ вокруг bbox в пикселях
    padding: int = 8
    # Дополнительный отступ снизу — ценник часто выступает ниже bbox
    padding_bottom_extra: int = 40

    # Минимальный размер кропа для OCR.
    # Если ценник меньше — апскейлим до этого размера.
    min_ocr_width: int = 200
    min_ocr_height: int = 80

    # Максимальный размер — ограничиваем чтобы OCR не тормозил на огромных кропах
    max_ocr_width: int = 1200
    max_ocr_height: int = 600

    # Применять ли автоматическую коррекцию перспективы
    apply_perspective: bool = True

    # Применять ли дополнительный CLAHE после кропа
    apply_clahe: bool = True

    # Применять ли повышение резкости (unsharp mask)
    apply_sharpen: bool = True

    # Сохранять ли промежуточные кропы для отладки
    debug_save: bool = False
    debug_dir: str = "debug_crops"


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def _safe_crop(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    Вырезает область из кадра с padding, не выходя за границы.
    Возвращает None если область вырождена.
    """
    h, w = frame.shape[:2]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def _find_tag_corners(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Пытается найти 4 угла ценника внутри кропа через контурный анализ.
    Возвращает массив 4 точек (TL, TR, BR, BL) или None если не нашли.

    Логика:
    1. Переводим в серый + адаптивный порог
    2. Ищем контуры
    3. Берём самый большой контур с 4 углами (аппроксимация)
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Адаптивный порог лучше работает при неравномерном освещении
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )

    # Morphology чтобы замкнуть разорванные края
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    h, w = crop.shape[:2]
    crop_area = w * h

    # Ищем контур похожий на прямоугольник
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Слишком маленький или слишком большой (весь кроп) — пропускаем
        if area < crop_area * 0.15 or area > crop_area * 0.98:
            continue

        # Аппроксимируем контур до многоугольника
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Нам нужен четырёхугольник
        if len(approx) != 4:
            continue

        if area > best_area:
            best_area = area
            best = approx

    if best is None:
        return None

    # Упорядочиваем точки: TL, TR, BR, BL
    pts = best.reshape(4, 2).astype(np.float32)
    return _order_points(pts)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки в порядке: TL, TR, BR, BL.
    Используется для perspective transform.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: минимальная сумма x+y
    rect[2] = pts[np.argmax(s)]   # BR: максимальная сумма x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR: минимальная разность y-x
    rect[3] = pts[np.argmax(diff)]  # BL: максимальная разность y-x

    return rect


def _perspective_transform(
    crop: np.ndarray,
    corners: np.ndarray
) -> np.ndarray:
    """
    Применяет perspective warp по 4 найденным углам.
    Результат — прямоугольное выровненное изображение ценника.
    """
    tl, tr, br, bl = corners

    # Вычисляем размер выходного изображения
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))

    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    if width <= 0 or height <= 0:
        return crop

    dst = np.array([
        [0,         0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0,         height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(crop, M, (width, height))
    return warped


def _upscale(
    img: np.ndarray,
    min_w: int,
    min_h: int,
    max_w: int,
    max_h: int
) -> np.ndarray:
    """
    Апскейлит изображение если оно меньше минимального размера.
    Ограничивает до максимального размера.
    Сохраняет пропорции.
    """
    h, w = img.shape[:2]

    # Апскейл если слишком маленький
    if w < min_w or h < min_h:
        scale = max(min_w / w, min_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # INTER_CUBIC лучше сохраняет текст при апскейле
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w

    # Даунскейл если слишком большой
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE на L-канал LAB для улучшения контраста текста."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _sharpen(img: np.ndarray) -> np.ndarray:
    """
    Unsharp mask — повышает резкость текста.
    Особенно полезно после апскейла (INTER_CUBIC размывает края).
    """
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharp


# ─────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────

class Preprocessor:
    """
    Подготавливает вырезанный ценник для OCR.

    Пример:
        preprocessor = Preprocessor(config)
        for frame, ts_ms in loader.frames():
            for det in detector.detect(frame):
                crop = preprocessor.process(frame, det)
                if crop is not None:
                    # crop готов для ocr_engine
                    text_results = ocr.read(crop)
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()

        if self.config.debug_save:
            from pathlib import Path
            Path(self.config.debug_dir).mkdir(exist_ok=True)

        self._debug_counter = 0

    def process(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> Optional[np.ndarray]:
        """
        Полный пайплайн обработки одного ценника.

        Args:
            frame:     исходный кадр BGR
            detection: детекция от PriceTagDetector

        Returns:
            BGR изображение готовое для OCR, или None если не удалось обработать
        """
        cfg = self.config

        # ── Шаг 1: Вырезаем bbox с padding ───────────────────────────────
        crop = _safe_crop(
            frame,
            detection.x_min, detection.y_min,
            detection.x_max, detection.y_max + cfg.padding_bottom_extra,
            padding=cfg.padding,
        )
        if crop is None or crop.size == 0:
            logger.debug("Пустой кроп, пропускаем")
            return None

        # ── Шаг 2: Коррекция перспективы ─────────────────────────────────
        if cfg.apply_perspective:
            corners = _find_tag_corners(crop)
            if corners is not None:
                warped = _perspective_transform(crop, corners)
                # Проверяем что результат разумный (не схлопнулся)
                wh, ww = warped.shape[:2]
                ch, cw = crop.shape[:2]
                if ww > cw * 0.3 and wh > ch * 0.3:
                    crop = warped
                    logger.debug("Перспектива скорректирована")
                else:
                    logger.debug("Warp дал слишком маленький результат, оставляем исходный кроп")
            else:
                logger.debug("Углы не найдены, перспектива не корректируется")

        # ── Шаг 3: Поворот вертикальных ценников ────────────────────────
        # Ценники на видео обычно стоят вертикально (height > width * 1.3).
        # Порог 1.3 — не трогаем почти квадратные и горизонтальные кропы.
        ch, cw = crop.shape[:2]
        if ch > cw * 1.3:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ch, cw = cw, ch  # обновляем размеры после поворота

        # ── Шаг 4: Даунскейл если кроп слишком большой для OCR ───────────
        # При 4K видео кроп и так достаточно крупный.
        # Апскейл не делаем — он добавляет артефакты.
        # Даунскейл только если превышает максимум.
        if cw > cfg.max_ocr_width or ch > cfg.max_ocr_height:
            scale = min(cfg.max_ocr_width / cw, cfg.max_ocr_height / ch)
            new_w = int(cw * scale)
            new_h = int(ch * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ── Шаг 5: Лёгкий CLAHE только если кроп тёмный ─────────────────
        if cfg.apply_clahe:
            # Проверяем среднюю яркость — если нормальная, не трогаем
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(gray.mean())
            if mean_brightness < 100:  # тёмный кроп — улучшаем
                crop = _apply_clahe(crop)

        # ── Отладка ──────────────────────────────────────────────────────
        if cfg.debug_save:
            self._save_debug(crop, detection)

        return crop

    def process_batch(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[tuple[Detection, np.ndarray]]:
        """
        Обрабатывает список детекций за один вызов.
        Возвращает список (detection, crop) только для успешных.
        """
        results = []
        for det in detections:
            crop = self.process(frame, det)
            if crop is not None:
                results.append((det, crop))
        return results

    def _save_debug(self, crop: np.ndarray, det: Detection):
        """Сохраняет кроп для ручного контроля качества."""
        from pathlib import Path
        path = Path(self.config.debug_dir) / f"crop_{self._debug_counter:04d}_{det.color}.jpg"
        cv2.imwrite(str(path), crop)
        self._debug_counter += 1


# ─────────────────────────────────────────────
# CLI для отладки
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Использование:")
        print("  python preprocessor.py <video.mp4>")
        print("  python preprocessor.py <image.jpg>")
        print("  python preprocessor.py <папка_с_кадрами/>")
        print()
        print("  Кропы сохраняются в debug_crops/<имя_источника>/")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Конфиги
    try:
        from config import LOADER_CONFIG, DETECTOR_CONFIG, PREPROCESSOR_CONFIG
        print("Конфиг загружен из config.py")
    except (ImportError, AttributeError):
        from video_loader import VideoLoaderConfig
        from detector import DetectorConfig
        LOADER_CONFIG = VideoLoaderConfig(fps_sample=1.0, apply_clahe=True)
        DETECTOR_CONFIG = DetectorConfig(use_hsv_fallback=True)
        PREPROCESSOR_CONFIG = PreprocessorConfig()
        print("Используются дефолтные настройки")

    # Всегда включаем debug_save при запуске CLI
    PREPROCESSOR_CONFIG.debug_save = True

    # Папка для кропов: debug_crops/<имя_источника>/
    source_name = input_path.stem if not input_path.is_dir() else input_path.name
    output_dir = Path("debug_crops") / source_name
    output_dir.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_CONFIG.debug_dir = str(output_dir)

    from video_loader import VideoLoader
    from detector import PriceTagDetector

    detector    = PriceTagDetector(DETECTOR_CONFIG)
    preprocessor = Preprocessor(PREPROCESSOR_CONFIG)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    # ── Режим: папка с кадрами ────────────────────────────────────────────
    if input_path.is_dir():
        frames_paths = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ])
        if not frames_paths:
            print(f"В папке {input_path} не найдено изображений")
            sys.exit(1)

        print(f"\nНайдено кадров: {len(frames_paths)}")
        print(f"Кропы → {output_dir}/\n")

        total_dets  = 0
        total_crops = 0

        for i, fp in enumerate(frames_paths, 1):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            dets    = detector.detect(frame)
            results = preprocessor.process_batch(frame, dets)

            total_dets  += len(dets)
            total_crops += len(results)

            if len(dets) > 0:
                print(f"  [{i:3d}/{len(frames_paths)}] {fp.name}: "
                      f"{len(dets)} детекций → {len(results)} кропов")

        print(f"\nИтого: {total_dets} детекций, {total_crops} кропов")
        print(f"Сохранены в: {output_dir}/")

    # ── Режим: одно изображение ───────────────────────────────────────────
    elif input_path.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(input_path))
        if frame is None:
            print(f"Не удалось открыть: {input_path}")
            sys.exit(1)

        dets    = detector.detect(frame)
        results = preprocessor.process_batch(frame, dets)

        print(f"Найдено: {len(dets)} ценников, обработано: {len(results)}")
        for i, (det, crop) in enumerate(results):
            print(f"  [{i+1:2d}] {det.color:8s} {det.width:4d}x{det.height:<4d}px "
                  f"→ crop {crop.shape[1]}x{crop.shape[0]}px")
        print(f"\nСохранены в: {output_dir}/")

    # ── Режим: видеофайл ──────────────────────────────────────────────────
    else:
        loader = VideoLoader(input_path, LOADER_CONFIG)
        info   = loader.get_video_info()

        print(f"\nВидео: {info['filename']}  "
              f"{info['width']}x{info['height']}  "
              f"{info['fps']:.1f}fps  {info['duration_sec']:.1f}с")
        print(f"Кропы → {output_dir}/\n")

        total_dets  = 0
        total_crops = 0
        frame_num   = 0

        for frame, ts_ms in loader.frames():
            frame_num  += 1
            dets        = detector.detect(frame)
            results     = preprocessor.process_batch(frame, dets)
            total_dets  += len(dets)
            total_crops += len(results)

            if len(dets) > 0:
                print(f"  [{frame_num:4d}] t={ts_ms/1000:6.2f}с  "
                      f"{len(dets)} детекций → {len(results)} кропов")

        print(f"\nКадров обработано: {frame_num}")
        print(f"Итого: {total_dets} детекций, {total_crops} кропов")
        print(f"Сохранены в: {output_dir}/")
