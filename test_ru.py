"""test_ru.py — тест русской модели PaddleOCR"""
import cv2
import sys
from paddleocr import PaddleOCR

img_path = sys.argv[1] if len(sys.argv) > 1 else "debug_crops/25_2-10/crop_0021_orange.jpg"
img = cv2.imread(img_path)

print(f"Загружаем модель lang='ru'...")
ocr = PaddleOCR(lang='ru', use_angle_cls=True, use_gpu=False, show_log=False)

print(f"Читаем: {img_path}")
result = ocr.ocr(img, cls=True)

print(f"\nБлоки:")
if result and result[0]:
    for line in result[0]:
        bbox, (text, conf) = line
        print(f"  conf={conf:.2f}  '{text}'")
else:
    print("  Блоков не найдено")
