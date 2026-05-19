"""
video_loader.py — Глава 1: Загрузка и предобработка видео
==========================================================
Отвечает за:
  - Извлечение кадров из видеофайла с нужным шагом
  - Fisheye-коррекцию (опциональная, если известны параметры камеры)
  - Адаптивное улучшение контраста (CLAHE)
  - Фильтрацию слишком похожих кадров (по SSIM / разности)
  - Возврат итератора (frame, timestamp_ms) для дальнейшей обработки
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Настройки по умолчанию
# ─────────────────────────────────────────────

@dataclass
class VideoLoaderConfig:
    # Сколько кадров в секунду брать (None = все кадры)
    fps_sample: float = 2.0

    # Минимальная разность между соседними кадрами (0–1).
    # Кадры, похожие более чем на (1 - diff_threshold), пропускаются.
    diff_threshold: float = 0.02

    # Применять ли CLAHE для улучшения контраста
    apply_clahe: bool = True

    # Применять ли fisheye-коррекцию
    apply_undistort: bool = False

    # Параметры камеры для fisheye-коррекции.
    # Если None и apply_undistort=True — используется эвристика.
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None

    # Масштаб кадра перед обработкой (1.0 = оригинал)
    # Полезно для ускорения на больших видео
    scale: float = 1.0

    # Размер кернела для лёгкого шумоподавления (0 = выключено)
    denoise_ksize: int = 3


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def _build_clahe(clip_limit: float = 2.0, tile_size: int = 8) -> cv2.CLAHE:
    """Создаёт объект CLAHE для L-канала в LAB-пространстве."""
    return cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )


def apply_clahe(frame: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """
    Применяет CLAHE к L-каналу изображения в LAB-пространстве.
    Улучшает контраст без пересветки — важно для съёмки в магазине
    со смешанным освещением.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _default_fisheye_params(frame_w: int, frame_h: int):
    """
    Эвристические параметры дисторсии для широкоугольной камеры робота.
    Подходит как стартовая точка если нет калибровочных данных.
    Для точной коррекции нужна калибровка через шахматную доску.
    """
    cx, cy = frame_w / 2, frame_h / 2
    fx = fy = frame_w * 0.7   # focal length ~ 0.7 * width (широкий угол)

    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)

    # k1, k2 — радиальная дисторсия (fisheye даёт сильный k1)
    # p1, p2 — тангенциальная дисторсия
    dist_coeffs = np.array([-0.35, 0.15, 0.0, 0.0, -0.05], dtype=np.float64)

    return camera_matrix, dist_coeffs


def undistort_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Корректирует fisheye-дисторсию кадра.
    alpha=0 — обрезает края (нет чёрных полос)
    alpha=1 — сохраняет все пиксели (могут быть чёрные углы)
    """
    h, w = frame.shape[:2]
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha
    )
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cam_mtx)

    # Обрезаем до валидной области если alpha < 1
    if alpha < 1.0:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y+rh, x:x+rw]
            undistorted = cv2.resize(undistorted, (w, h))  # возвращаем оригинальный размер

    return undistorted


def _frame_diff(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    Вычисляет долю изменившихся пикселей между двумя кадрами.
    Используется для пропуска дублирующихся кадров при стоянке робота.
    Возвращает значение от 0 (идентичные) до 1 (полностью разные).
    """
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    # Нормируем на максимальное возможное изменение
    return float(np.mean(diff)) / 255.0


def _soft_denoise(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Лёгкое размытие для подавления шума без потери текста."""
    if ksize <= 1:
        return frame
    # Bilateral сохраняет края — важно для текста ценников
    return cv2.bilateralFilter(frame, ksize, 75, 75)


# ─────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────

class VideoLoader:
    """
    Загружает видео и выдаёт предобработанные кадры.

    Пример использования:
        loader = VideoLoader("robot_video.mp4")
        for frame, timestamp_ms in loader.frames():
            # frame — BGR numpy array, готовый для детектора
            # timestamp_ms — позиция в видео в миллисекундах
            process(frame, timestamp_ms)
    """

    def __init__(
        self,
        video_path: str | Path,
        config: Optional[VideoLoaderConfig] = None
    ):
        self.video_path = Path(video_path)
        self.config = config or VideoLoaderConfig()

        if not self.video_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {self.video_path}")

        self._clahe = _build_clahe() if self.config.apply_clahe else None
        self._camera_matrix = None
        self._dist_coeffs = None
        self._prev_frame: Optional[np.ndarray] = None

        logger.info(f"VideoLoader готов: {self.video_path.name}")

    # ── публичные методы ──────────────────────

    def frames(self) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Генератор кадров.
        Yields: (frame_bgr, timestamp_ms)
        """
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {self.video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / video_fps

            # Шаг между извлекаемыми кадрами
            if self.config.fps_sample and self.config.fps_sample < video_fps:
                frame_step = max(1, int(video_fps / self.config.fps_sample))
            else:
                frame_step = 1

            logger.info(
                f"Видео: {video_fps:.1f} fps, {total_frames} кадров, "
                f"{duration_sec:.1f}с → берём каждый {frame_step}-й кадр"
            )

            # Инициализируем параметры undistort при первом кадре
            frame_idx = 0
            yielded = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Пропускаем кадры по шагу
                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                # Масштабирование
                if self.config.scale != 1.0:
                    frame = self._resize(frame, self.config.scale)

                # Fisheye-коррекция
                if self.config.apply_undistort:
                    frame = self._undistort(frame)

                # Шумоподавление
                if self.config.denoise_ksize > 1:
                    frame = _soft_denoise(frame, self.config.denoise_ksize)

                # CLAHE
                if self._clahe is not None:
                    frame = apply_clahe(frame, self._clahe)

                # Фильтрация похожих кадров
                if self._is_duplicate(frame):
                    frame_idx += 1
                    continue

                self._prev_frame = frame.copy()
                yield frame, timestamp_ms
                yielded += 1
                frame_idx += 1

            logger.info(f"Извлечено {yielded} кадров из {frame_idx} всего")

        finally:
            cap.release()

    def get_video_info(self) -> dict:
        """Возвращает метаданные видео."""
        cap = cv2.VideoCapture(str(self.video_path))
        info = {
            "filename": self.video_path.name,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 25),
        }
        cap.release()
        return info

    # ── приватные методы ──────────────────────

    def _resize(self, frame: np.ndarray, scale: float) -> np.ndarray:
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        """Применяет коррекцию дисторсии, инициализируя параметры при первом вызове."""
        if self._camera_matrix is None:
            h, w = frame.shape[:2]
            if self.config.camera_matrix is not None:
                self._camera_matrix = self.config.camera_matrix
                self._dist_coeffs = self.config.dist_coeffs
            else:
                logger.warning("Параметры камеры не заданы — используется эвристика fisheye")
                self._camera_matrix, self._dist_coeffs = _default_fisheye_params(w, h)

        return undistort_frame(frame, self._camera_matrix, self._dist_coeffs, alpha=0.5)

    def _is_duplicate(self, frame: np.ndarray) -> bool:
        """Возвращает True если кадр слишком похож на предыдущий."""
        if self._prev_frame is None:
            return False
        diff = _frame_diff(self._prev_frame, frame)
        return diff < self.config.diff_threshold


# ─────────────────────────────────────────────
# CLI для быстрого тестирования
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python video_loader.py <path_to_video> [output_dir]")
        print("  Пример: python video_loader.py robot_01.mp4")
        print("  Пример: python video_loader.py robot_01.mp4 D:/frames")
        print()
        print("  Кадры: <output_dir>/<video_stem>/frame_XXXXXXXX.jpg")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    base_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("frames_output")

    # Подпапка = имя видео без расширения
    # robot_01.mp4  ->  frames_output/robot_01/
    # hall_dairy.avi -> frames_output/hall_dairy/
    output_dir = base_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Конфиг из config.py если есть, иначе дефолт
    try:
        from config import LOADER_CONFIG as config
        print("  Конфиг загружен из config.py")
    except ImportError:
        config = VideoLoaderConfig(
            fps_sample=1.0,
            apply_clahe=True,
            apply_undistort=False,
            diff_threshold=0.02,
        )
        print("  config.py не найден — используются дефолтные настройки")

    loader = VideoLoader(video_path, config)
    info = loader.get_video_info()

    print(f"\n Video: {info['filename']}")
    print(f"   Разрешение: {info['width']}x{info['height']}")
    print(f"   FPS: {info['fps']:.1f}, длительность: {info['duration_sec']:.1f}с")
    print(f"   Кадров всего: {info['total_frames']}")
    print(f"   Сохраняем в: {output_dir}/\n")

    saved = 0
    for frame, ts_ms in loader.frames():
        out_path = output_dir / f"frame_{ts_ms:08d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1
        if saved % 10 == 0:
            print(f"  Сохранено {saved} кадров...")

    print(f"\nГотово! Сохранено {saved} кадров в {output_dir}/")
