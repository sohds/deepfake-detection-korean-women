import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class DeepFakeDataset(Dataset):
    def __init__(self, dataframe, transform=None, study_type='face'):
        self.data = dataframe
        self.transform = transform
        self.study_type = study_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        def safe_open_image(path, key):
            try:
                if pd.isna(path) or not os.path.exists(path):
                    print(f"Warning: {key} image not found at {path}")
                    return torch.zeros((3, 299, 299))
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img
            except Exception as e:
                print(f"Error loading {key} image at {path}: {str(e)}")
                return torch.zeros((3, 299, 299))

        result = {}
        
        if self.study_type == 'face' or self.study_type == 'face_nose' or self.study_type == 'face_central' or self.study_type == 'face_cheeks_nose':
            result['face'] = safe_open_image(row['face'], 'face')
        
        if self.study_type == 'face_nose' or self.study_type == 'face_cheeks_nose':
            result['nose'] = safe_open_image(row['nose'], 'nose')
        
        if self.study_type == 'face_central':
            result['central'] = safe_open_image(row['central'], 'central')
        
        if self.study_type == 'face_cheeks_nose':
            result['left_cheek'] = safe_open_image(row['left_cheek'], 'left_cheek')
            result['right_cheek'] = safe_open_image(row['right_cheek'], 'right_cheek')

        result['label'] = torch.tensor(1 if row['deepfake'] else 0, dtype=torch.long)
        
        return result