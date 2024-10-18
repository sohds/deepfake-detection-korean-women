from PIL import Image
import torch
from torch.utils.data import Dataset

# Custom Dataset
class DeepFakeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        face = Image.open(row['face'])
        nose = Image.open(row['nose'])
        central = Image.open(row['central'])
        left_cheek = Image.open(row['left_cheek'])
        right_cheek = Image.open(row['right_cheek'])

        if self.transform:
            face = self.transform(face)
            nose = self.transform(nose)
            central = self.transform(central)
            left_cheek = self.transform(left_cheek)
            right_cheek = self.transform(right_cheek)

        label = torch.tensor(1 if row['deepfake'] else 0, dtype=torch.long)
        
        return {
            'face': face,
            'nose': nose,
            'central': central,
            'left_cheek': left_cheek,
            'right_cheek': right_cheek,
            'label': label
        }