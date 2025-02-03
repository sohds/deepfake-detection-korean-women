import csv
import cv2
import torch
import torch.nn as nn
# from utils.landmark_extract import get_face_masks
from utils.normalize import preprocess_image
from utils.gradcam_overlay import explain
from utils.evaluate import calculate_presence_binary, calculate_area_in_mask

def load_model(model_path, device):
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2).to(device)
    model.eval()
    return model

def process_images(image_paths, model, device, detector, predictor, landmark_indices, output_csv, presence_binary_csv):
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
                image_prep = preprocess_image(image).to(device)
                salience_map = explain(image_prep, model, device)
                
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
