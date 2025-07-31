import cv2
import numpy as np
import os

save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extracted_plates"))
os.makedirs(save_dir, exist_ok=True)

image_list = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "img", f"car_0{i}.jpg")) for i in range(1, 6)]

pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)
current_image_index = 0
draw = None
img = None

color_map = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)]  # TL, TR, BR, BL 색
label_map = ["TL", "TR", "BR", "BL"]

def onMouse(event, x, y, flags, param):
    global pts_cnt, pts, draw, img, current_image_index

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x, y), 8, (255, 0, 255), -1)
        cv2.putText(draw, str(pts_cnt + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.imshow("License Plate Extractor", draw)

        pts[pts_cnt] = [x, y]
        pts_cnt += 1

        if pts_cnt == 4:
            print("[INFO] 4개의 꼭짓점 입력 완료. 번호판 추출 시작.")

            sm = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            topLeft = pts[np.argmin(sm)]
            bottomRight = pts[np.argmax(sm)]
            topRight = pts[np.argmin(diff)]
            bottomLeft = pts[np.argmax(diff)]

            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # ✅ 정렬된 점 시각적으로 표시
            corners = [topLeft, topRight, bottomRight, bottomLeft]
            for idx, pt in enumerate(corners):
                pt = tuple(pt.astype(int))
                cv2.circle(draw, pt, 10, color_map[idx], -1)
                cv2.putText(draw, label_map[idx], (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map[idx], 2)
            cv2.imshow("License Plate Extractor", draw)

            w1 = np.linalg.norm(bottomRight - bottomLeft)
            w2 = np.linalg.norm(topRight - topLeft)
            h1 = np.linalg.norm(topRight - bottomRight)
            h2 = np.linalg.norm(topLeft - bottomLeft)
            width = int(max(w1, w2))
            height = int(max(h1, h2))

            pts2 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])

            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, mtrx, (width, height))

            filename = os.path.join(save_dir, f"plate_0{current_image_index+1}.jpg")
            cv2.imwrite(filename, result)
            print(f"> 번호판 이미지 저장 완료: {filename}")

            current_image_index += 1
            if current_image_index < len(image_list):
                load_next_image()
            else:
                print("✅ 모든 번호판 이미지 저장 완료.")
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

def load_next_image():
    global draw, img, pts_cnt, pts
    img_path = image_list[current_image_index]
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 이미지 불러오기 실패: {img_path}")
        return
    draw = img.copy()
    pts_cnt = 0
    pts = np.zeros((4, 2), dtype=np.float32)
    cv2.imshow("License Plate Extractor", draw)
    print(f"[INFO] {img_path} 불러오기 완료. 번호판 꼭짓점 4개를 순서 없이 클릭하세요.")

load_next_image()
cv2.setMouseCallback("License Plate Extractor", onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()