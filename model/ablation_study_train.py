import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from timm import create_model
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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

# Feature Extractor using XceptionNet
class XceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super(XceptionFeatureExtractor, self).__init__()
        self.xception = create_model('xception', pretrained=True)
        self.xception.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        return self.xception(x)

# Swin Transformer Classifier
class SwinTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(SwinTransformerClassifier, self).__init__()
        self.swin = create_model('swin_base_patch4_window7_224', pretrained=True)
        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)

    def forward(self, x):
        return self.swin(x)

# Full Model for each ablation study
class DeepFakeDetectionModel(nn.Module):
    def __init__(self, study_type):
        super(DeepFakeDetectionModel, self).__init__()
        self.study_type = study_type
        self.face_extractor = XceptionFeatureExtractor()
        self.secondary_extractor = XceptionFeatureExtractor()
        
        if study_type == 'face_nose':
            input_dim = 4096  # 2048 (face) + 2048 (nose)
        elif study_type == 'face_central':
            input_dim = 4096  # 2048 (face) + 2048 (central)
        else:  # face_cheeks_nose
            input_dim = 8192  # 2048 (face) + 2048 (left_cheek) + 2048 (nose) + 2048 (right_cheek)
        
        self.classifier = SwinTransformerClassifier(input_dim)

    def forward(self, face, secondary):
        face_features = self.face_extractor(face)
        secondary_features = self.secondary_extractor(secondary)
        
        if self.study_type == 'face_cheeks_nose':
            combined_features = torch.cat((face_features, secondary_features[:, :2048], 
                                           secondary_features[:, 2048:4096], 
                                           secondary_features[:, 4096:]), dim=1)
        else:
            combined_features = torch.cat((face_features, secondary_features), dim=1)
        
        return self.classifier(combined_features)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training function with best checkpoint saving (including epoch and optimizer state)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, start_epoch=0, study_type='face_nose'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_acc = 0.0  # Initialize best accuracy to track the best checkpoint

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs} [Training]"):
            face = batch['face'].to(device)

            if study_type == 'face_nose':
                secondary = batch['nose'].to(device)
            elif study_type == 'face_central':
                secondary = batch['central'].to(device)
            else:  # face_cheeks_nose
                secondary = torch.cat((batch['left_cheek'], batch['nose'], batch['right_cheek']), dim=1).to(device)

            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(face, secondary)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs} [Validation]"):
                face = batch['face'].to(device)

                if study_type == 'face_nose':
                    secondary = batch['nose'].to(device)
                elif study_type == 'face_central':
                    secondary = batch['central'].to(device)
                else:  # face_cheeks_nose
                    secondary = torch.cat((batch['left_cheek'], batch['nose'], batch['right_cheek']), dim=1).to(device)

                labels = batch['label'].to(device)
                outputs = model(face, secondary)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save the best checkpoint with epoch and optimizer state
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, f'best_checkpoint_{study_type}.pth')
            print(f"New best model saved with accuracy: {best_acc:.4f}, Epoch: {epoch+1}")

    print(f"Training complete. Best accuracy: {best_acc:.4f}")

# Loading the model from checkpoint for additional training
def load_checkpoint(model, optimizer, filename='best_checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    print(f"Loaded checkpoint from epoch {start_epoch}, with best accuracy: {best_acc:.4f}")
    return model, optimizer, start_epoch

# Main execution with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepFake Detection Training')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file containing metadata')
    parser.add_argument('--study_type', type=str, choices=['face_nose', 'face_central', 'face_cheeks_nose'], required=True, help='Ablation study type to use')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from the best checkpoint')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # Load dataset and split into train, validation, and test sets
    df = pd.read_csv(args.csv)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = DeepFakeDataset(train_df, transform=transform)
    val_dataset = DeepFakeDataset(val_df, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    model = DeepFakeDetectionModel(args.study_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Resume training if specified
    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, filename=f'best_checkpoint_{args.study_type}.pth')

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs, start_epoch=start_epoch, study_type=args.study_type)