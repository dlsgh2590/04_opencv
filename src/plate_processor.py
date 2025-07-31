#새로운 번호판 이미지 처리 파이프라인 파일
# 번호판 이미지 전처리 파이프라인
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# [1] 번호판 이미지 로드 함수
def load_extracted_plate(plate_name):
    plate_path = f'../extracted_plates/{plate_name}.jpg'
    if os.path.exists(plate_path):
        plate_img = cv2.imread(plate_path)
        print(f"번호판 이미지 로드 완료: {plate_img.shape}")
        return plate_img
    else:
        print(f"파일을 찾을 수 없습니다: {plate_path}")
        return None

# [2] 그레이스케일 변환 함수
def convert_to_grayscale(plate_img):
    return cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

# [3] 명암 대비 최대화 함수
def maximize_contrast(gray_plate):
    # 탑햇과 블랙햇 연산을 통해 문자 대비 향상
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    tophat = cv2.morphologyEx(gray_plate, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray_plate, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray_plate, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)
    # 히스토그램 평활화로 전체 명암 균일화
    return cv2.equalizeHist(enhanced)

# [4] 적응형 임계값과 Otsu 이진화
def adaptive_threshold_plate(enhanced_plate):
    blurred = cv2.GaussianBlur(enhanced_plate, (3, 3), 0)  # 노이즈 제거
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_adaptive, thresh_otsu

# [5] 윤곽선 탐지 함수
def find_contours_in_plate(thresh_plate):
    contours, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 윤곽선을 컬러 이미지로 시각화
    contour_image = cv2.cvtColor(thresh_plate, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contours, contour_image

# [6] 처리된 결과 이미지 저장 함수
def save_processed_results(plate_name, gray, enhanced, thresh, contour_img):
    save_dir = '../processed_plates'
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(f'{save_dir}/{plate_name}_1_gray.png', gray)
    cv2.imwrite(f'{save_dir}/{plate_name}_2_enhanced.png', enhanced)
    cv2.imwrite(f'{save_dir}/{plate_name}_3_thresh.png', thresh)
    cv2.imwrite(f'{save_dir}/{plate_name}_4_contours.png', contour_img)
    print(f"{plate_name} 처리 이미지 저장 완료")

# [7] 문자 추출 가능성 평가 (윤곽선 수 기반)
def prepare_for_next_step(contours):
    count = len(contours)
    if count < 5:
        print("윤곽선이 너무 적습니다.")
    elif count > 20:
        print("윤곽선이 너무 많습니다. 노이즈 가능성 있음.")
    else:
        print("적절한 윤곽선 수입니다.")

    # 적절한 크기의 윤곽선만 문자 후보로 추정
    possible_chars = sum(1 for cnt in contours if 30 < cv2.contourArea(cnt) < 2000)
    print(f"잠재적 문자 후보: {possible_chars}개")
    return possible_chars

# [8] 전체 전처리 파이프라인 함수
def process_extracted_plate(plate_name):
    print(f"===== {plate_name} 처리 시작 =====")

    # (1) 번호판 이미지 로드
    plate = load_extracted_plate(plate_name)
    if plate is None:
        return None

    # (2) 전처리 단계
    gray = convert_to_grayscale(plate)
    enhanced = maximize_contrast(gray)
    thresh, _ = adaptive_threshold_plate(enhanced)
    contours, contour_img = find_contours_in_plate(thresh)

    # (3) 결과 저장 및 문자 후보 평가
    save_processed_results(plate_name, gray, enhanced, thresh, contour_img)
    possible_chars = prepare_for_next_step(contours)

    print(f"===== {plate_name} 처리 완료 =====")
    return {
        'original': plate,
        'gray': gray,
        'enhanced': enhanced,
        'threshold': thresh,
        'contours': len(contours),
        'potential_chars': possible_chars,
        'contour_result': contour_img
    }

# [9] 폴더 내 모든 번호판 이미지 일괄 처리
def batch_process_plates():
    plate_dir = '../extracted_plates'
    if not os.path.exists(plate_dir):
        print(f"폴더 없음: {plate_dir}")
        return {}

    # .png 확장자 이미지만 대상으로 처리
    files = [f for f in os.listdir(plate_dir) if f.endswith('.png')]
    if not files:
        print("번호판 이미지 없음")
        return {}

    results = {}
    for file in files:
        name = file.rsplit('.', 1)[0]  # 확장자 제거
        result = process_extracted_plate(name)
        if result:
            results[name] = result

    print(f"총 {len(results)}개 번호판 처리 완료")
    return results

# [10] 단일 이미지 테스트 실행
if __name__ == '__main__':
    process_extracted_plate('plate_02')