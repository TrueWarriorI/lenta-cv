"""
debug_ocr.py — показывает сырые OCR блоки для одного кропа
Использование: python debug_ocr.py <crop.jpg>
"""
import sys
import cv2
import logging
logging.basicConfig(level=logging.WARNING)

if len(sys.argv) < 2:
    print("Использование: python debug_ocr.py <crop.jpg>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    print(f"Не удалось открыть: {sys.argv[1]}")
    sys.exit(1)

print(f"Размер кропа: {img.shape[1]}x{img.shape[0]}px")

# Инициализируем PaddleOCR
from paddleocr import PaddleOCR
import inspect

sig = inspect.signature(PaddleOCR.__init__).parameters
kwargs = {}
if "lang" in sig: kwargs["lang"] = "ru"  # русская кириллическая модель
if "use_angle_cls" in sig: kwargs["use_angle_cls"] = True
if "use_textline_orientation" in sig: kwargs["use_textline_orientation"] = True
if "use_gpu" in sig: kwargs["use_gpu"] = False
if "show_log" in sig: kwargs["show_log"] = False

ocr = PaddleOCR(**kwargs)
raw = ocr.ocr(img, cls=True)

print(f"\nВсего блоков: {len(raw[0]) if raw and raw[0] else 0}")
print(f"{'─'*60}")

if raw and raw[0]:
    for i, line in enumerate(raw[0]):
        bbox_pts, (text, conf) = line
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        h = y2 - y1
        w = x2 - x1
        print(f"  [{i+1:2d}] conf={conf:.2f}  h={h:3d}px  pos=({x1},{y1})  '{text}'")
else:
    print("  Блоков не найдено!")

# Также пробуем QR через cv2
print(f"\n{'─'*60}")
print("QR детектор (cv2):")
qr = cv2.QRCodeDetector()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for label, variant in [
    ("оригинал", gray),
    ("увеличенный x2", cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)),
    ("инвертированный", cv2.bitwise_not(gray)),
]:
    data, points, _ = qr.detectAndDecode(variant)
    if data:
        print(f"  ✅ {label}: {data}")
    else:
        print(f"  ❌ {label}: не прочитан")

# Пробуем EDSR улучшение
print(f"\nQR через EDSR:")
try:
    from ocr_engine import _enhance_qr_zone
    enhanced = _enhance_qr_zone(img)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape)==3 else enhanced

    # Сохраняем для проверки
    cv2.imwrite("debug_qr_enhanced.jpg", enhanced)
    print(f"  Сохранён: debug_qr_enhanced.jpg")

    rotations = [
        ("без поворота",     enhanced_gray),
        ("90° по часовой",   cv2.rotate(enhanced_gray, cv2.ROTATE_90_CLOCKWISE)),
        ("90° против часов", cv2.rotate(enhanced_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("180°",             cv2.rotate(enhanced_gray, cv2.ROTATE_180)),
        ("инвертированный",  cv2.bitwise_not(enhanced_gray)),
    ]
    for label, variant in rotations:
        data, points, _ = qr.detectAndDecode(variant)
        if data:
            print(f"  ✅ EDSR {label}: {data[:80]}")
        else:
            print(f"  ❌ EDSR {label}: не прочитан")
except Exception as e:
    print(f"  ❌ EDSR ошибка: {e}")
