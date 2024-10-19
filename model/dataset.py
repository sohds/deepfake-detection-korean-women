import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import pandas as pd

class DeepFakeDataset(Dataset):
    def __init__(self, dataframe, study_type='face'):
        self.data = dataframe
        self.study_type = study_type
        
        # Define transforms for Swin Transformer (face-only) and Xception
        self.swin_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.xception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        def safe_open_image(path, key, transform):
            try:
                if pd.isna(path) or not os.path.exists(path):
                    print(f"Warning: {key} image not found at {path}")
                    return torch.zeros((3, 224, 224) if transform == self.swin_transform else (3, 299, 299))
                img = Image.open(path).convert('RGB')
                img = transform(img)
                return img
            except Exception as e:
                print(f"Error loading {key} image at {path}: {str(e)}")
                return torch.zeros((3, 224, 224) if transform == self.swin_transform else (3, 299, 299))

        result = {}
        
        if self.study_type == 'face':
            result['face'] = safe_open_image(row['face'], 'face', self.swin_transform)
        else:
            result['face'] = safe_open_image(row['face'], 'face', self.xception_transform)
            if self.study_type in ['face_nose', 'face_cheeks_nose']:
                result['nose'] = safe_open_image(row['nose'], 'nose', self.xception_transform)
            if self.study_type == 'face_central':
                result['central'] = safe_open_image(row['central'], 'central', self.xception_transform)
            if self.study_type == 'face_cheeks_nose':
                result['left_cheek'] = safe_open_image(row['left_cheek'], 'left_cheek', self.xception_transform)
                result['right_cheek'] = safe_open_image(row['right_cheek'], 'right_cheek', self.xception_transform)

        result['label'] = torch.tensor(1 if row['deepfake'] else 0, dtype=torch.long)
        
        return result