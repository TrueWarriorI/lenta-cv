"""
app.py — Глава 6: Streamlit веб-интерфейс
==========================================
Загрузка видео → детекция → OCR → трекинг → скачивание CSV
"""

import streamlit as st
import cv2
import tempfile
import time
import json
import pandas as pd
from pathlib import Path
from tracker import ProductMatcher

# ─────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Lenta Tech — Распознавание ценников",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ Lenta Tech — Распознавание ценников")
st.caption("Загрузите видео с робота → получите CSV с данными всех ценников")

# ─────────────────────────────────────────────
# Загрузка модулей (с кешированием)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Загружаем модели...")
def load_pipeline():
    """Загружает все модели один раз при старте."""
    from config import LOADER_CONFIG, DETECTOR_CONFIG, PREPROCESSOR_CONFIG, OCR_CONFIG, TRACKER_CONFIG
    from video_loader  import VideoLoader
    from detector      import PriceTagDetector
    from preprocessor  import Preprocessor
    from ocr_engine    import OCREngine
    from tracker       import PriceTagTracker

    detector     = PriceTagDetector(DETECTOR_CONFIG)
    preprocessor = Preprocessor(PREPROCESSOR_CONFIG)
    ocr_engine   = OCREngine(OCR_CONFIG)

    return {
        "loader_config":     LOADER_CONFIG,
        "tracker_config":    TRACKER_CONFIG,
        "detector":          detector,
        "preprocessor":      preprocessor,
        "ocr_engine":        ocr_engine,
        "VideoLoader":       VideoLoader,
        "PriceTagTracker":   PriceTagTracker,
    }


# ─────────────────────────────────────────────
# Боковая панель — настройки
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Настройки")

    fps_sample = st.slider(
        "Кадров в секунду для анализа",
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="Меньше = быстрее, но можно пропустить ценники"
    )

    min_confidence = st.slider(
        "Минимальная уверенность детектора",
        min_value=0.1, max_value=0.9, value=0.35, step=0.05,
    )

    show_preview = st.checkbox("Показывать превью кадров", value=True)

    st.divider()
    st.markdown("**О проекте**")
    st.markdown(
        "Система компьютерного зрения для распознавания "
        "ценников с видеопотока робота. Lenta Tech Life Hack."
    )

# ─────────────────────────────────────────────
# Основной контент
# ─────────────────────────────────────────────

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Загрузите видео с робота",
        type=["mp4", "avi", "mov", "mkv"],
        help="Видео со стеллажами магазина, снятое роботом. Рекомендуемый размер: до 200 MB"
    )

with col2:
    if uploaded_file:
        st.success(f"✅ Загружено: **{uploaded_file.name}**")
        file_size_mb = uploaded_file.size / 1024 / 1024
        st.info(f"Размер: {file_size_mb:.1f} MB")

# ─────────────────────────────────────────────
# Обработка видео
# ─────────────────────────────────────────────

if uploaded_file is not None:
    if st.button("🚀 Запустить распознавание", type="primary", use_container_width=True):

        # Сохраняем загруженный файл во временную папку
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        try:
            pipeline = load_pipeline()

            # Обновляем настройки из слайдеров
            pipeline["loader_config"].fps_sample = fps_sample
            pipeline["detector"].config.confidence = min_confidence

            VideoLoader   = pipeline["VideoLoader"]
            PriceTagTracker = pipeline["PriceTagTracker"]
            detector      = pipeline["detector"]
            preprocessor  = pipeline["preprocessor"]
            ocr_engine    = pipeline["ocr_engine"]

            loader  = VideoLoader(tmp_path, pipeline["loader_config"])
            tracker = PriceTagTracker(pipeline["tracker_config"])

            # Метаданные видео
            info = loader.get_video_info()

            st.markdown("---")
            st.subheader("📊 Прогресс обработки")

            meta_cols = st.columns(4)
            meta_cols[0].metric("Файл", info["filename"])
            meta_cols[1].metric("Разрешение", f"{info['width']}×{info['height']}")
            meta_cols[2].metric("Длительность", f"{info['duration_sec']:.0f}с")
            meta_cols[3].metric("FPS видео", f"{info['fps']:.0f}")

            # Прогресс-бар
            progress_bar  = st.progress(0, text="Начинаем обработку...")
            status_text   = st.empty()
            preview_place = st.empty()
            stats_place   = st.empty()

            # Счётчики
            frame_count   = 0
            total_dets    = 0
            total_crops   = 0
            start_time    = time.time()

            total_frames_est = int(info["total_frames"] / max(1, info["fps"] / fps_sample))

            for frame, ts_ms in loader.frames():
                frame_count += 1

                # Прогресс
                progress = min(frame_count / max(total_frames_est, 1), 0.99)
                elapsed  = time.time() - start_time
                eta      = (elapsed / frame_count) * (total_frames_est - frame_count) if frame_count > 0 else 0

                progress_bar.progress(
                    progress,
                    text=f"Кадр {frame_count} | "
                         f"Детекций: {total_dets} | "
                         f"Прошло: {elapsed:.0f}с | "
                         f"Осталось: ~{eta:.0f}с"
                )

                # Детекция
                detections = detector.detect(frame)
                total_dets += len(detections)

                # Превью каждые 10 кадров
                if show_preview and frame_count % 10 == 1 and detections:
                    vis = detector.draw(frame, detections)
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    # Уменьшаем для отображения
                    h, w = vis_rgb.shape[:2]
                    scale = min(800 / w, 400 / h)
                    vis_small = cv2.resize(
                        vis_rgb,
                        (int(w * scale), int(h * scale))
                    )
                    preview_place.image(
                        vis_small,
                        caption=f"Кадр {frame_count} | t={ts_ms/1000:.1f}с | {len(detections)} ценников",
                        use_container_width=True,
                    )

                # Препроцессинг + OCR + трекинг
                for det in detections:
                    crop = preprocessor.process(frame, det)
                    if crop is None:
                        continue

                    total_crops += 1

                    ocr_result = ocr_engine.read(
                        crop,
                        color=det.color,
                        filename=uploaded_file.name,
                        frame_timestamp=ts_ms,
                        x_min=det.x_min, y_min=det.y_min,
                        x_max=det.x_max, y_max=det.y_max,
                    )

                    tracker.update(det, ocr_result, ts_ms)

                # Статистика
                stats_place.markdown(
                    f"**Обработано кадров:** {frame_count} | "
                    f"**Детекций:** {total_dets} | "
                    f"**Кропов:** {total_crops} | "
                    f"**Треков:** {len(tracker._tracks)}"
                )

            # Финализация
            progress_bar.progress(1.0, text="Финализация...")
            records = tracker.finalize()

            # Обогащение через базу товаров если есть GT CSV
            matcher = ProductMatcher(csv_dir="csv")
            video_stem = Path(uploaded_file.name).stem
            if matcher.load_for_video(video_stem):
                records = [matcher.match(r) for r in records]
                st.info(f"✅ Названия товаров уточнены через базу данных")

            # Сохраняем CSV
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            csv_path = output_dir / f"{Path(uploaded_file.name).stem}_results.csv"
            tracker.export_csv(records, csv_path)

            # Читаем CSV для отображения
            df = pd.read_csv(csv_path, encoding="utf-8-sig")

            progress_bar.progress(1.0, text="✅ Готово!")

            # ── Результаты ────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("✅ Результаты распознавания")

            res_cols = st.columns(4)
            res_cols[0].metric("Уникальных ценников", len(records))
            res_cols[1].metric("С ценой", df["price_card"].notna().sum())
            res_cols[2].metric("С названием", (df["product_name"] != "").sum())
            res_cols[3].metric("Время обработки", f"{time.time()-start_time:.0f}с")

            # Таблица результатов
            display_cols = [
                "filename", "color", "price_card", "price_default",
                "discount_amount", "product_name", "barcode", "frame_timestamp"
            ]
            st.dataframe(
                df[[c for c in display_cols if c in df.columns]],
                use_container_width=True,
                height=400,
            )

            # Кнопка скачивания CSV
            with open(csv_path, "rb") as f:
                csv_bytes = f.read()

            st.download_button(
                label="⬇️ Скачать CSV",
                data=csv_bytes,
                file_name=csv_path.name,
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            # Удаляем временный файл
            try:
                tmp_path.unlink()
            except Exception:
                pass

# ─────────────────────────────────────────────
# Инструкция если видео не загружено
# ─────────────────────────────────────────────

else:
    st.markdown("---")
    st.markdown("""
    ### Как использовать

    1. **Загрузите видео** — MP4, AVI, MOV или MKV с записью робота
    2. **Настройте параметры** в боковой панели (опционально)
    3. **Нажмите «Запустить»** — система обработает видео
    4. **Скачайте CSV** с результатами распознавания

    ### Что распознаётся

    | Поле | Описание |
    |------|----------|
    | `price_card` | Цена по карте лояльности |
    | `price_default` | Цена без карты |
    | `discount_amount` | Размер скидки |
    | `product_name` | Название товара |
    | `barcode` | Штрихкод |
    | `color` | Цвет ценника |
    | `frame_timestamp` | Время в видео (мс) |
    """)
