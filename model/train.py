import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataset import DeepFakeDataset
from model import DeepFakeDetectionModel
import os

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
            labels = batch['label'].to(device)

            if study_type == 'face':
                outputs = model(face)
            elif study_type == 'face_nose':
                nose = batch['nose'].to(device)
                outputs = model(face, nose)
            elif study_type == 'face_central':
                central = batch['central'].to(device)
                outputs = model(face, central)
            elif study_type == 'face_cheeks_nose':
                nose = batch['nose'].to(device)
                left_cheek = batch['left_cheek'].to(device)
                right_cheek = batch['right_cheek'].to(device)
                secondary = torch.cat((left_cheek, nose, right_cheek), dim=1)
                outputs = model(face, secondary)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
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
                labels = batch['label'].to(device)

                if study_type == 'face':
                    secondary = None
                elif study_type == 'face_nose':
                    secondary = batch['nose'].to(device)
                elif study_type == 'face_central':
                    secondary = batch['central'].to(device)
                else:  # face_cheeks_nose
                    secondary = torch.cat((batch['left_cheek'], batch['nose'], batch['right_cheek']), dim=1).to(device)

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
            torch.save(checkpoint, os.path.join(args.output_dir, f'{study_type}_best_checkpoint_{epoch+1}.pth'))
            print(f"New best model saved with accuracy: {best_acc:.4f}, Epoch: {epoch+1}")

    print(f"Training complete. Best accuracy: {best_acc:.4f}")

# Loading the model from checkpoint for additional training
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    print(f"Loaded checkpoint from epoch {start_epoch}, with best accuracy: {best_acc:.4f}")
    return model, optimizer, start_epoch

# Main execution with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepFake Detection Model Training')
    parser.add_argument('--csv', '-f', type=str, default='/content/drive/MyDrive/Capstone-Design/multiscaleDetect/meta_data.csv', required=True, help='Path CSV Files neeeded for training')
    parser.add_argument('--study_type', '-t', type=str, choices=['face', 'face_nose', 'face_central', 'face_cheeks_nose'], required=True, help='Ablation study type to use')
    parser.add_argument('--resume', '-r', type=bool, default=False, help='Resume training from the best checkpoint')
    parser.add_argument('--checkpoint', '-c', type=str, default='', help='checkpoint file to resume training')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--output_dir', '-o', type=str, default='/content/drive/MyDrive/Capstone-Design/multiscaleDetect/checkpoints', help='Output directory to save ROC curve plots')
    args = parser.parse_args()

    # Load dataset and split into train, validation, and test sets
    df = pd.read_csv(args.csv)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print('Dataset Loaded.')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = DeepFakeDataset(train_df, transform=transform, study_type=args.study_type)
    val_dataset = DeepFakeDataset(val_df, transform=transform, study_type=args.study_type)
    print('Datasets Created.')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    print('Train Loader Created.')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print('Validation Loader Created.')

    # num_classes, Model, Loss, Optimizer
    num_classes = 2
    model = DeepFakeDetectionModel(args.study_type, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Resume training if specified
    start_epoch = 0
    if args.resume:
        if not args.checkpoint:
            print('Please provide a checkpoint file to resume training.')
        if os.path.isfile(args.checkpoint):
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
        else:
            print(f"No checkpoint found at {args.checkpoint}, starting from scratch.")

    print('Training Started.')
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs, start_epoch=start_epoch, study_type=args.study_type)