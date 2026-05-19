"""
debug_qr.py — тестирует pyzbar на кропе ценника
Использование: python debug_qr.py <crop.jpg>
"""
import sys
import cv2

if len(sys.argv) < 2:
    print("Использование: python debug_qr.py <crop.jpg>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    print(f"Не удалось открыть: {sys.argv[1]}")
    sys.exit(1)

print(f"Размер: {img.shape[1]}x{img.shape[0]}px")

from pyzbar import pyzbar
from ocr_engine import _enhance_qr_zone

enhanced = _enhance_qr_zone(img)
h, w = img.shape[:2]

variants = [
    ("оригинал полный",       img),
    ("enhanced зона",         enhanced),
    ("enhanced rot90cw",      cv2.rotate(enhanced, cv2.ROTATE_90_CLOCKWISE)),
    ("enhanced rot90ccw",     cv2.rotate(enhanced, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ("enhanced rot180",       cv2.rotate(enhanced, cv2.ROTATE_180)),
    ("нижний правый 50%",     img[int(h*0.5):, int(w*0.5):]),
    ("нижний правый rot90cw", cv2.rotate(img[int(h*0.5):, int(w*0.5):], cv2.ROTATE_90_CLOCKWISE)),
]

found = False
for name, variant in variants:
    if variant is None or variant.size == 0:
        continue
    gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if len(variant.shape)==3 else variant
    # Пробуем оригинал и увеличенный
    for scale_name, img_to_try in [("", gray), (" x2", cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2)))]:
        result = pyzbar.decode(img_to_try)
        if result:
            print(f"✅ {name}{scale_name}: {result[0].data.decode()}")
            found = True
            break

if not found:
    print("❌ QR не прочитан ни в одном варианте")
    print("\nВозможные причины:")
    print("  1. QR слишком размыт из-за движения камеры")
    print("  2. Нужен лучший кадр — найди кроп где робот стоял неподвижно")
