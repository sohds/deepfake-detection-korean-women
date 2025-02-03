import cv2
import numpy as np


def extract_face_regions(image, landmarks, landmark_indices):
    """
    주요 얼굴 영역에 대한 마스크 추출 함수

    추출 영역:
      - eyebrow: 눈썹 영역 (left_eyebrow + right_eyebrow)
      - eye: 눈 영역 (left_eye + right_eye)
      - nose: 코 영역
      - mouth: 입 영역
      - chin: 턱 영역
      - cheek: 볼 영역 (left_cheek + right_cheek)
      - forehead: 이마 영역

    매개변수:
      image           : NumPy 배열 형태의 입력 이미지
      landmarks       : dlib 얼굴 랜드마크 (68)
      landmark_indices: 얼굴 부위별 landmark 인덱스 딕셔너리
                        "left_eyebrow", "right_eyebrow",
                        "left_eye", "right_eye",
                        "nose",
                        "mouth",
                        "chin",
                        "left_cheek", "right_cheek" 등의 키가 있어야 함

    반환:
      각 얼굴 영역에 대한 바이너리 마스크를 담은 딕셔너리
      (key: 'eyebrow', 'eye', 'nose', 'mouth', 'chin', 'cheek', 'forehead')
    """

    # 이미지의 높이, 너비를 구하기 (이미지 shape가 HxW 또는 HxWxC라고 가정)
    h, w = image.shape[:2]
    
    # 각 부분에 대한 빈 마스크 생성 (단일 채널)
    left_eyebrow_mask = np.zeros((h, w), dtype=np.uint8)
    right_eyebrow_mask = np.zeros((h, w), dtype=np.uint8)
    left_eye_mask = np.zeros((h, w), dtype=np.uint8)
    right_eye_mask = np.zeros((h, w), dtype=np.uint8)
    nose_mask = np.zeros((h, w), dtype=np.uint8)
    mouth_mask = np.zeros((h, w), dtype=np.uint8)
    chin_mask = np.zeros((h, w), dtype=np.uint8)
    cheek_mask = np.zeros((h, w), dtype=np.uint8)
    forehead_mask = np.zeros((h, w), dtype=np.uint8)
    
    # ------------------- eyebrow 추출 -------------------
    left_eyebrow_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["left_eyebrow"]],
        dtype=np.int32
    )
    cv2.fillPoly(left_eyebrow_mask, [left_eyebrow_points], 255)

    right_eyebrow_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["right_eyebrow"]],
        dtype=np.int32
    )
    cv2.fillPoly(right_eyebrow_mask, [right_eyebrow_points], 255)
    
    # left_eyebrow + right_eyebrow
    eyebrow_mask = cv2.bitwise_or(left_eyebrow_mask, right_eyebrow_mask)
    
    # ------------------- eye 추출 -------------------
    left_eye_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["left_eye"]],
        dtype=np.int32
    )
    cv2.fillPoly(left_eye_mask, [left_eye_points], 255)

    right_eye_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["right_eye"]],
        dtype=np.int32
    )
    cv2.fillPoly(right_eye_mask, [right_eye_points], 255)
    
    # left_eye + right_eye
    eye_mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
    
    # ------------------- nose 추출 -------------------
    nose_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["nose"]],
        dtype=np.int32
    )
    cv2.fillPoly(nose_mask, [nose_points], 255)

    # ------------------- mouth 추출 -------------------
    mouth_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["mouth"]],
        dtype=np.int32
    )
    cv2.fillPoly(mouth_mask, [mouth_points], 255)
    
    # ------------------- chin 추출 -------------------
    chin_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["chin"]],
        dtype=np.int32
    )
    cv2.fillPoly(chin_mask, [chin_points], 255)
    
    # ------------------- cheek 추출 -------------------
    left_cheek_mask = np.zeros((h, w), dtype=np.uint8)
    right_cheek_mask = np.zeros((h, w), dtype=np.uint8)

    left_cheek_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["left_cheek"]],
        dtype=np.int32
    )
    cv2.fillPoly(left_cheek_mask, [left_cheek_points], 255)

    right_cheek_points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices["right_cheek"]],
        dtype=np.int32
    )
    cv2.fillPoly(right_cheek_mask, [right_cheek_points], 255)

    # left_cheek + right_cheek
    cheek_mask = cv2.bitwise_or(left_cheek_mask, right_cheek_mask)

    
    # ------------------- forehead 추출 -------------------
    # 코와 눈썹의 위치를 기준으로 이마 영역을 추정
    forehead_point1 = (nose_points[0][0], nose_points[0][1] - 100)
    forehead_point2 = (left_eyebrow_points[3][0], left_eyebrow_points[3][1] - 60)
    forehead_point3 = (left_eyebrow_points[0][0], left_eyebrow_points[0][1] - 40)
    forehead_point4 = (right_eyebrow_points[-1][0], right_eyebrow_points[-1][1] - 40)
    forehead_point5 = (right_eyebrow_points[-3][0], right_eyebrow_points[-3][1] - 60)

    forehead_points = np.array(
        [forehead_point1, forehead_point2, forehead_point3] +
        left_eyebrow_points.tolist() +
        right_eyebrow_points.tolist() +
        [forehead_point4, forehead_point5],
        dtype=np.int32
    )
    cv2.fillPoly(forehead_mask, [forehead_points], 255)
    

    # 각 영역의 마스크를 담은 딕셔너리 반환
    return {
        "eyebrow": eyebrow_mask,
        "eye": eye_mask,
        "nose": nose_mask,
        "mouth": mouth_mask,
        "chin": chin_mask,
        "cheek": cheek_mask,
        "forehead": forehead_mask
    }
