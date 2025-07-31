import cv2
import numpy as np
import os

# 저장 디렉토리 설정
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extracted_plates"))
os.makedirs(save_dir, exist_ok=True)

# 이미지 목록 구성
image_list = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "img", f"car_0{i}.jpg")) for i in range(1, 6)]

def maximize_contrast(gray):
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((3, 3), np.uint8))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((3, 3), np.uint8))
    add = cv2.add(gray, tophat)
    subtract = cv2.subtract(add, blackhat)
    return cv2.equalizeHist(subtract)

def binarize_image(gray):
    max_contrast = maximize_contrast(gray)

    adaptive = cv2.adaptiveThreshold(max_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 19, 9)
    _, otsu = cv2.threshold(max_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return adaptive, otsu