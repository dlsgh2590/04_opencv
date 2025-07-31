import cv2
import numpy as np
import os

# 저장 디렉토리 설정
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extracted_plates"))
os.makedirs(save_dir, exist_ok=True)

# 이미지 목록 구성
image_list = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "img", f"car_0{i}.jpg")) for i in range(1, 6)]