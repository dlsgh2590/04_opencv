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

def find_and_draw_contours(img, binary, save_path_prefix):
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.copy(img)
    cv2.drawContours(temp_result, contours, -1, (255, 255, 0), 1)

    height, width, channel = img.shape
    img_result = np.zeros((height, width, channel), dtype=np.uint8)

    candidate_cnt = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        rate = w / h if h != 0 else 0

        if 0.25 < rate < 1.0 and 20 < w < 200 and 20 < h < 200:
            cv2.rectangle(temp_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = img[y:y+h, x:x+w]
            img_result[y:y+h, x:x+w] = roi
            candidate_cnt += 1

    print(f"[INFO] 후보 개수: {candidate_cnt}")
    cv2.imwrite(f"{save_path_prefix}_contours.png", temp_result)
    cv2.imwrite(f"{save_path_prefix}_candidates.png", img_result)
