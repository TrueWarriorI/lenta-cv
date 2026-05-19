"""
train_detector.py — Дообучение YOLOv8 на ценниках Ленты
=========================================================
Использует ground truth CSV разметку (поставляется вместе с видео)
для создания датасета и дообучения YOLOv8.

Структура входных данных:
    Labeled/
    ├── 25_2-10/
    │   ├── 25_2-10.mp4
    │   └── 25_2-10.csv
    ├── 26_3-15/
    │   ├── 26_3-15.mp4
    │   └── 26_3-15.csv
    └── ...

Использование:
    python train_detector.py
    python train_detector.py --labeled Labeled --epochs 50 --model yolov8n.pt
"""

import os
import cv2
import re
import random
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────

def _parse_coord(value) -> float:
    """Парсит координату с запятой или точкой как разделителем."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        match = re.search(r"[\d.]+", value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return 0.0


def _parse_timestamp(value) -> int:
    """Парсит timestamp в мс."""
    try:
        return int(float(str(value).replace(",", ".")))
    except (ValueError, TypeError):
        return 0


def _to_yolo(x_min, y_min, x_max, y_max, img_w, img_h) -> tuple:
    """
    Конвертирует абсолютные координаты bbox в нормализованный YOLO формат.
    Возвращает (cx, cy, bw, bh) — все в диапазоне [0, 1].
    """
    x_min = max(0.0, min(img_w, x_min))
    x_max = max(0.0, min(img_w, x_max))
    y_min = max(0.0, min(img_h, y_min))
    y_max = max(0.0, min(img_h, y_max))

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, bw)),
        max(0.0, min(1.0, bh)),
    )


def _load_csv(csv_path: Path) -> list[dict]:
    """
    Загружает GT CSV и возвращает список записей.
    Автоматически определяет разделитель.
    """
    import csv

    for sep in [",", ";", "\t", "|"]:
        try:
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=sep)
                rows = list(reader)
                if rows and len(rows[0]) > 1:
                    logger.info(f"  CSV загружен (разделитель='{sep}'), строк: {len(rows)}")
                    return rows
        except Exception:
            continue

    return []


def _find_datasets(labeled_root: Path) -> list[dict]:
    """
    Находит все папки с видео + CSV в labeled_root.
    Возвращает список словарей {name, video, csv}.
    """
    datasets = []

    if not labeled_root.exists():
        logger.error(f"Папка не найдена: {labeled_root}")
        return datasets

    for folder in sorted(labeled_root.iterdir()):
        if not folder.is_dir():
            continue

        videos = list(folder.glob("*.mp4")) + list(folder.glob("*.avi")) + list(folder.glob("*.mov"))
        csvs   = list(folder.glob("*.csv"))

        if videos and csvs:
            datasets.append({
                "name":  folder.name,
                "video": videos[0],
                "csv":   csvs[0],
            })
            logger.info(f"✅ Датасет: {folder.name} | видео: {videos[0].name} | csv: {csvs[0].name}")
        else:
            logger.warning(f"⚠️  Пропущено: {folder.name} — нет видео или CSV")

    return datasets


# ─────────────────────────────────────────────
# Создание датасета
# ─────────────────────────────────────────────

def build_dataset(
    labeled_root: Path,
    output_dir: Path,
    val_split: float = 0.2,
    padding: int = 0,
) -> int:
    """
    Создаёт YOLO-датасет из размеченных видео.

    Args:
        labeled_root: папка с подпапками (видео + CSV)
        output_dir:   куда сохранять датасет
        val_split:    доля валидационных кадров
        padding:      отступ вокруг bbox в пикселях (0 = не добавлять)

    Returns:
        Количество сохранённых кадров
    """
    # Создаём структуру папок
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    datasets = _find_datasets(labeled_root)
    if not datasets:
        logger.error("Датасеты не найдены!")
        return 0

    frame_counter   = 0
    total_annotations = 0

    for ds in datasets:
        logger.info(f"\n📂 Обрабатываем: {ds['name']}")

        rows = _load_csv(ds["csv"])
        if not rows:
            logger.warning(f"  Пустой CSV, пропускаем")
            continue

        # Парсим координаты и timestamp
        parsed = []
        for row in rows:
            try:
                ts    = _parse_timestamp(row.get("frame_timestamp", "0"))
                x_min = _parse_coord(row.get("x_min", "0"))
                y_min = _parse_coord(row.get("y_min", "0"))
                x_max = _parse_coord(row.get("x_max", "0"))
                y_max = _parse_coord(row.get("y_max", "0"))

                if x_max <= x_min or y_max <= y_min:
                    continue

                parsed.append({
                    "ts":    ts,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "color": row.get("color", ""),
                })
            except Exception as e:
                logger.debug(f"  Ошибка строки: {e}")

        logger.info(f"  Валидных аннотаций: {len(parsed)}")

        # Группируем по timestamp
        from collections import defaultdict
        grouped = defaultdict(list)
        for p in parsed:
            grouped[p["ts"]].append(p)

        logger.info(f"  Уникальных кадров: {len(grouped)}")

        # Открываем видео
        cap = cv2.VideoCapture(str(ds["video"]))
        if not cap.isOpened():
            logger.error(f"  Не удалось открыть: {ds['video']}")
            continue

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"  Видео: {n_frames} кадров, {fps:.1f} fps")

        processed = 0

        for ts_ms, annotations in tqdm(
            grouped.items(),
            desc=f"  {ds['name']}",
            leave=False,
        ):
            # Переходим к кадру по timestamp в мс
            cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
            ret, frame = cap.read()

            if not ret or frame is None:
                logger.debug(f"  Не удалось прочитать кадр t={ts_ms}мс")
                continue

            img_h, img_w = frame.shape[:2]
            labels = []

            for ann in annotations:
                x_min = ann["x_min"] - padding
                y_min = ann["y_min"] - padding
                x_max = ann["x_max"] + padding
                y_max = ann["y_max"] + padding

                cx, cy, bw, bh = _to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

                # Фильтруем слишком маленькие bbox
                if bw < 0.005 or bh < 0.005:
                    continue

                # Класс 0 = price_tag (один класс для всех типов ценников)
                labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                continue

            # Train / val split
            split = "val" if random.random() < val_split else "train"

            # Сохраняем кадр
            img_name   = f"frame_{frame_counter:08d}.jpg"
            img_path   = output_dir / "images" / split / img_name
            label_path = output_dir / "labels" / split / img_name.replace(".jpg", ".txt")

            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            with open(label_path, "w") as f:
                f.write("\n".join(labels))

            frame_counter   += 1
            total_annotations += len(labels)
            processed       += 1

        cap.release()
        logger.info(f"  Сохранено: {processed} кадров, {total_annotations} аннотаций")

    # Создаём dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""# YOLO датасет — ценники Ленты
path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: price_tag
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info(f"\n📄 dataset.yaml сохранён: {yaml_path}")

    # Статистика
    train_n = len(list((output_dir / "images" / "train").glob("*.jpg")))
    val_n   = len(list((output_dir / "images" / "val").glob("*.jpg")))

    logger.info(f"\n{'='*50}")
    logger.info(f"✅ ДАТАСЕТ ГОТОВ")
    logger.info(f"   Всего кадров:     {frame_counter}")
    logger.info(f"   Всего аннотаций:  {total_annotations}")
    logger.info(f"   Train:            {train_n}")
    logger.info(f"   Val:              {val_n}")
    logger.info(f"{'='*50}")

    return frame_counter


# ─────────────────────────────────────────────
# Обучение YOLOv8
# ─────────────────────────────────────────────

def train_yolo(
    dataset_yaml: Path,
    base_model: str = "yolov8n.pt",
    output_dir: Path = Path("trained_models"),
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
) -> Path:
    """
    Дообучает YOLOv8 на созданном датасете.

    Args:
        dataset_yaml: путь к dataset.yaml
        base_model:   базовая модель (yolov8n.pt / yolov8s.pt)
        output_dir:   куда сохранять обученную модель
        epochs:       количество эпох
        imgsz:        размер изображения
        batch:        размер батча (уменьши если мало RAM)
        device:       'cpu' или '0' для GPU

    Returns:
        Путь к лучшим весам (best.pt)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Установи ultralytics: pip install ultralytics")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*50}")
    logger.info(f"🚀 НАЧИНАЕМ ОБУЧЕНИЕ YOLOv8")
    logger.info(f"   Базовая модель: {base_model}")
    logger.info(f"   Эпох: {epochs}")
    logger.info(f"   Размер: {imgsz}px")
    logger.info(f"   Батч: {batch}")
    logger.info(f"   Устройство: {device}")
    logger.info(f"{'='*50}\n")

    model = YOLO(base_model)

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir),
        name="lenta_price_tags",
        exist_ok=True,
        # Аугментации — важны для разных условий освещения
        hsv_h=0.015,   # вариация оттенка
        hsv_s=0.3,     # вариация насыщенности
        hsv_v=0.3,     # вариация яркости
        fliplr=0.0,    # не отражаем горизонтально (ценники не симметричны)
        degrees=5.0,   # небольшой поворот
        translate=0.1, # смещение
        scale=0.3,     # масштабирование
    )

    # Путь к лучшим весам
    # YOLO сохраняет в runs/detect/<project>/<name>/weights/
    best_pt = Path("runs") / "detect" / output_dir.name / "lenta_price_tags" / "weights" / "best.pt"

    if best_pt.exists():
        logger.info(f"\n✅ Обучение завершено!")
        logger.info(f"   Лучшая модель: {best_pt}")
        logger.info(f"\nДля использования в проекте добавь в config.py:")
        logger.info(f"   model_path='{best_pt}'")
    else:
        logger.warning("Модель не найдена по ожидаемому пути — проверь папку trained_models/")

    return best_pt


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Дообучение YOLOv8 на ценниках Ленты"
    )
    parser.add_argument(
        "--labeled", type=str, default="Labeled",
        help="Папка с размеченными видео (default: Labeled)"
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset",
        help="Папка для датасета (default: dataset)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Базовая модель YOLOv8 (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Количество эпох (default: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Размер изображения (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Размер батча (default: 8, уменьши при нехватке RAM)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Устройство: cpu или 0 для GPU (default: cpu)"
    )
    parser.add_argument(
        "--skip-dataset", action="store_true",
        help="Пропустить создание датасета (если уже создан)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Только создать датасет, не обучать"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Доля валидационных данных (default: 0.2)"
    )

    args = parser.parse_args()

    labeled_root = Path(args.labeled)
    dataset_dir  = Path(args.dataset)
    output_dir   = Path("trained_models")

    # Шаг 1 — Создаём датасет
    if not args.skip_dataset:
        n = build_dataset(
            labeled_root=labeled_root,
            output_dir=dataset_dir,
            val_split=args.val_split,
        )
        if n == 0:
            print("\n❌ Датасет пуст — проверь структуру папки Labeled/")
            exit(1)
    else:
        logger.info("Пропускаем создание датасета (--skip-dataset)")

    # Шаг 2 — Обучаем YOLOv8
    if not args.skip_train:
        yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            print(f"\n❌ dataset.yaml не найден: {yaml_path}")
            exit(1)

        best_model = train_yolo(
            dataset_yaml=yaml_path,
            base_model=args.model,
            output_dir=output_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    else:
        logger.info("Пропускаем обучение (--skip-train)")
