import os
import cv2
import dlib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# utils 및 dataset에서 모듈 import
from utils.config import load_config
from utils.process import process_images
from utils.extract_landmark import extract_face_regions
from utils.normalize import preprocess_image
from utils.gradcam_overlay import explain, apply_cam_overlay
from utils.evaluate import calculate_presence_binary, calculate_area_in_mask
from visualization.graph_visualization import plot_images_bar, plot_area_bar
from visualization.gradcamsum import combine_and_enhance_images
from dataset.crop import get_face_landmarks, get_cropped_image

def process_single_image(image_path, model, device, detector, predictor, output_dir, landmark_indices):
    """단일 이미지 처리 함수"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None
    
    # 얼굴 감지
    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if len(faces) == 0:
        print(f"얼굴을 감지할 수 없습니다: {image_path}")
        return None
    
    # 랜드마크 추출 및 얼굴 영역 마스크 생성
    landmarks = predictor(image, faces[0])
    face_masks = extract_face_regions(image, landmarks, landmark_indices)
    
    # Grad-CAM 생성
    image_prep, image_var, visualize_image = preprocess_image(image)
    salience_map = explain(image_prep.to(device), model, device)
    visualization = apply_cam_overlay(image_var, visualize_image, model, device)
    
    # 결과 저장
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"gradcam_{output_filename}")
    cv2.imwrite(output_path, cv2.cvtColor((visualization * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # 각 얼굴 영역별 활성화 분석
    region_analysis = {}
    for region_name, mask in face_masks.items():
        presence = calculate_presence_binary(salience_map, mask)
        area, total_area = calculate_area_in_mask(salience_map, mask)
        region_analysis[region_name] = {
            'presence': presence,
            'area': area,
            'total_area': total_area,
            'area_ratio': area / total_area if total_area > 0 else 0
        }
    
    return {
        'image_path': image_path,
        'salience_map': salience_map,
        'visualization': visualization,
        'face_masks': face_masks,
        'region_analysis': region_analysis
    }

def save_evaluation_results(results, output_dir):
    """평가 결과를 CSV 파일로 저장"""
    area_results = []
    presence_results = []
    
    for result in results:
        area_row = {'image_path': result['image_path']}
        presence_row = {'image_path': result['image_path']}
        
        for region, analysis in result['region_analysis'].items():
            area_row[region] = analysis['area_ratio']
            presence_row[region] = analysis['presence']
        
        area_results.append(area_row)
        presence_results.append(presence_row)
    
    area_df = pd.DataFrame(area_results)
    presence_df = pd.DataFrame(presence_results)
    
    area_df.to_csv(os.path.join(output_dir, 'area_analysis.csv'), index=False)
    presence_df.to_csv(os.path.join(output_dir, 'presence_analysis.csv'), index=False)
    
    return area_df, presence_df

def preprocess_images(input_dir, output_dir):
    """이미지 전처리: 얼굴 감지 및 크롭"""
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    processed_paths = []
    for image_file in tqdm(image_files, desc="이미지 전처리 중"):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        # 얼굴 감지
        boxes = get_face_landmarks(image)
        if not boxes:
            print(f"얼굴을 감지할 수 없습니다: {image_path}")
            continue
        
        # 얼굴 영역 크롭
        cropped_face = get_cropped_image(image, boxes)
        if cropped_face is None:
            print(f"얼굴 크롭에 실패했습니다: {image_path}")
            continue
        
        # 크롭된 이미지 저장
        output_path = os.path.join(output_dir, f"cropped_{image_file}")
        cv2.imwrite(output_path, cropped_face)
        processed_paths.append(output_path)
    
    return processed_paths

def main():
    # 설정 로드
    config = load_config()
    
    # 디바이스 설정
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    checkpoint = torch.load(config['paths']['model_path'], map_location=device)
    model = checkpoint['model'].to(device)
    model.eval()
    
    # dlib 감지기 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config['paths']['face_detector'])
    
    # 랜드마크 인덱스 정의
    landmark_indices = {
        "left_eyebrow": range(17, 22),
        "right_eyebrow": range(22, 27),
        "left_eye": range(36, 42),
        "right_eye": range(42, 48),
        "nose": range(27, 36),
        "mouth": range(48, 68),
        "chin": [5, 6, 7, 8, 9, 10, 11, 54, 55, 56, 57, 58, 59, 48],
        "left_cheek": [0, 1, 2, 3, 4, 31, 32, 33],
        "right_cheek": [12, 13, 14, 15, 16, 33, 34, 35]
    }
    
    # 출력 디렉토리 생성
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # 이미지 전처리 (얼굴 감지 및 크롭)
    preprocessed_dir = os.path.join(config['paths']['output_dir'], 'preprocessed')
    processed_paths = preprocess_images(config['paths']['input_dir'], preprocessed_dir)
    
    # CSV 파일 경로 설정
    output_csv = os.path.join(config['paths']['output_dir'], 'area_analysis.csv')
    presence_binary_csv = os.path.join(config['paths']['output_dir'], 'presence_analysis.csv')
    
    # 이미지 처리 및 결과 저장
    process_images(
        image_paths=processed_paths,  # 전처리된 이미지 경로 사용
        model=model,
        device=device,
        detector=detector,
        predictor=predictor,
        landmark_indices=landmark_indices,
        output_csv=output_csv,
        presence_binary_csv=presence_binary_csv,
        output_dir=config['paths']['output_dir']
    )
    
    # 결과 시각화
    area_df = pd.read_csv(output_csv)
    presence_df = pd.read_csv(presence_binary_csv)
    
    plot_images_bar(presence_df.drop('Image Path', axis=1))
    plt.savefig(os.path.join(config['paths']['output_dir'], 'presence_analysis.png'))
    
    plot_area_bar(area_df.drop('Image Path', axis=1).mean().to_frame('Mean Values'),
                 area_df.columns[1:].tolist())
    plt.savefig(os.path.join(config['paths']['output_dir'], 'area_analysis.png'))
    
    # Grad-CAM 합성 이미지 생성
    combine_and_enhance_images(
        config['paths']['output_dir'],
        os.path.join(config['paths']['output_dir'], 'combined'),
        config.get('gradcam', {}).get('subfolder_prefix', 'gradcam'),
        scaling_factor=config.get('gradcam', {}).get('scaling_factor', 2)
    )
    
    print(f"처리 완료: 총 {len(processed_paths)}개의 이미지가 성공적으로 처리되었습니다.")

if __name__ == "__main__":
    main() 