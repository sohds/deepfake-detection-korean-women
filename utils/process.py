import os
import csv
import cv2
import torch
import torch.nn as nn
from utils.landmark_extract import get_face_masks
from utils.normalize import preprocess_image
from utils.gradcam_overlay import explain, apply_cam_overlay
from utils.evaluate import calculate_presence_binary, calculate_area_in_mask
from matplotlib import pyplot as plt

def load_model(model_path, device):
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2).to(device)
    model.eval()
    return model

def save_gradcam_image(visualization, image_path, output_dir):
    # 원본 이미지의 폴더 및 파일명 추출
    folder_name = os.path.basename(os.path.dirname(image_path))
    image_filename = os.path.basename(image_path)

    # 출력 폴더 생성
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # 저장할 파일 경로 설정 (파일명 앞에 `grad_cam_` 추가)
    output_path = os.path.join(output_folder, f"grad_cam_{image_filename}")

    # Grad-CAM 이미지 저장
    plt.imsave(output_path, visualization)
    print(f"Grad-CAM image saved at {output_path}")

def process_images(image_paths, model, device, detector, predictor, landmark_indices, output_csv, presence_binary_csv, output_dir):
    
    with open(output_csv, mode='w', newline='') as file1, open(presence_binary_csv, mode='w', newline='') as file2:
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)
        writer1.writerow(["Image Path"] + list(landmark_indices.keys()))
        writer2.writerow(["Image Path"] + list(landmark_indices.keys()))

        for image_path in image_paths:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                masks = get_face_masks(image, landmarks, landmark_indices)

                # 이미지 전처리
                image_prep, image_var, visualize_image = preprocess_image(image)

                # Grad-CAM 실행 및 결과 저장
                salience_map = explain(image_prep.to(device), model, device)
                visualization = apply_cam_overlay(image_var, visualize_image, model, device)

                # Grad-CAM 이미지 저장
                save_gradcam_image(visualization, image_path, output_dir)

                row1 = [image_path]
                row2 = [image_path]

                for mask in masks:
                    area, mask_area = calculate_area_in_mask(salience_map, mask)
                    activation_ratio = area / mask_area * 100 if mask_area > 0 else 0
                    presence_binary = calculate_presence_binary(salience_map, mask)
                    row1.append(round(activation_ratio, 4))
                    row2.append(presence_binary)

                writer1.writerow(row1)
                writer2.writerow(row2)