import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torchvision.transforms as transforms
import random

class DeepFakeDataset(Dataset):
    def __init__(self, dataframe, study_type='face', augment=True):
        self.data = dataframe
        self.study_type = study_type
        self.augment = augment
        
        # Define base transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

        # Define transforms for Swin Transformer (face-only) and Xception
        self.swin_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            self.base_transform
        ])
        self.xception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            self.base_transform
        ])

    def __len__(self):
        return len(self.data)

    def safe_open_image(self, path, key, transform):
        try:
            if pd.isna(path) or not os.path.exists(path):
                print(f"Warning: {key} image not found at {path}")
                return torch.zeros((3, 224, 224) if transform == self.swin_transform else (3, 299, 299))
            
            img = Image.open(path).convert('RGB')
            
            if self.augment:
                # Apply augmentation
                img = self.augment_transform(img)
                img = transform(img)
                img = self.add_gaussian_noise(img)
            else:
                # Apply only basic transformation
                img = transform(img)
            
            return img
        except Exception as e:
            print(f"Error loading {key} image at {path}: {str(e)}")
            return torch.zeros((3, 224, 224) if transform == self.swin_transform else (3, 299, 299))

    def add_gaussian_noise(self, img):
        noise = torch.randn_like(img) * 0.02
        return torch.clamp(img + noise, 0, 1)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        result = {}
        
        if self.study_type == 'face':
            result['face'] = self.safe_open_image(row['face'], 'face', self.swin_transform)
        else:
            result['face'] = self.safe_open_image(row['face'], 'face', self.xception_transform)
            if self.study_type in ['face_nose', 'face_cheeks_nose']:
                result['nose'] = self.safe_open_image(row['nose'], 'nose', self.xception_transform)
            if self.study_type == 'face_central':
                result['central'] = self.safe_open_image(row['central'], 'central', self.xception_transform)
            if self.study_type == 'face_cheeks_nose':
                result['left_cheek'] = self.safe_open_image(row['left_cheek'], 'left_cheek', self.xception_transform)
                result['right_cheek'] = self.safe_open_image(row['right_cheek'], 'right_cheek', self.xception_transform)

        result['label'] = torch.tensor(1 if row['deepfake'] else 0, dtype=torch.long)
        
        return result