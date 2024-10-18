from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

# Custom Dataset class for DeepFake dataset
class DeepFakeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        def safe_open_image(path):
            if pd.isna(path) or not os.path.exists(path):
                return None
            return Image.open(path)

        face = safe_open_image(row['face'])
        nose = safe_open_image(row['nose'])
        central = safe_open_image(row['central'])
        left_cheek = safe_open_image(row['left_cheek'])
        right_cheek = safe_open_image(row['right_cheek'])

        if self.transform:
            face = self.transform(face) if face else None
            nose = self.transform(nose) if nose else None
            central = self.transform(central) if central else None
            left_cheek = self.transform(left_cheek) if left_cheek else None
            right_cheek = self.transform(right_cheek) if right_cheek else None

        label = torch.tensor(1 if row['deepfake'] else 0, dtype=torch.long)
        
        return {
            'face': face,
            'nose': nose,
            'central': central,
            'left_cheek': left_cheek,
            'right_cheek': right_cheek,
            'label': label
        }