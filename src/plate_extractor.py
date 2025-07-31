# 자동차 번호판 추출 실습

import cv2
import numpy as np
import os

# ✅ 저장할 폴더 경로: 04_opencv/extracted_plates
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extracted_plates"))
os.makedirs(save_dir, exist_ok=True)

# 이미지 리스트
image_list = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "img", f"car_0{i}.jpg")) for i in range(1, 6)]

# 전역 변수
pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0
current_image_index = 0
draw = None
img = None

def onMouse(event, x, y, flags, param):
    global pts_cnt, pts, draw, img, current_image_index

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow("License Plate Extractor", draw)
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

        if pts_cnt == 4:
            sm = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            topLeft = pts[np.argmin(sm)]
            bottomRight = pts[np.argmax(sm)]
            topRight = pts[np.argmin(diff)]
            bottomLeft = pts[np.argmax(diff)]

            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            w1 = np.linalg.norm(bottomRight - bottomLeft)
            w2 = np.linalg.norm(topRight - topLeft)
            h1 = np.linalg.norm(topRight - bottomRight)
            h2 = np.linalg.norm(topLeft - bottomLeft)
            width = int(max(w1, w2))
            height = int(max(h1, h2))

            pts2 = np.float32([[0, 0], [width-1, 0],
                               [width-1, height-1], [0, height-1]])

            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, mtrx, (width, height))

            # 이미지 저장
            filename = os.path.join(save_dir, f"plate_0{current_image_index+1}.jpg")
            cv2.imwrite(filename, result)
            print(f"> 저장 완료: {filename}")

            # 다음 이미지로 전환
            current_image_index += 1
            if current_image_index < len(image_list):
                load_next_image()
            else:
                print("모든 이미지 저장 완료.")
                cv2.destroyAllWindows()

def load_next_image():
    global draw, img, pts_cnt, pts
    img_path = image_list[current_image_index]
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 불러오기 실패: {img_path}")
        return
    draw = img.copy()
    pts_cnt = 0
    pts = np.zeros((4, 2), dtype=np.float32)
    cv2.imshow("License Plate Extractor", draw)

# 첫 이미지 로딩 및 마우스 콜백 설정
load_next_image()
cv2.setMouseCallback("License Plate Extractor", onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()