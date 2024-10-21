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

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# Training function with best checkpoint saving (including epoch and optimizer state)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, start_epoch=0, study_type='face_nose'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint_dir = os.path.join(args.output_dir, args.saving_folder)
    createDirectory(checkpoint_dir)

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

            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")  # Debugging line
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
            train_acc = epoch_acc
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'{study_type}_best_checkpoint_{epoch+1}.pth'))
            print(f"New best model saved with accuracy: {best_acc:.4f}, Epoch: {epoch+1}")
        
        elif val_acc == best_acc:
            print(f"Model performance did not improve from the previous best accuracy: {best_acc:.4f}")
            print("Checking training accuracy for comparison.")
            if epoch_acc > train_acc:
                train_acc = epoch_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'{study_type}_best_checkpoint_{epoch+1}.pth'))
                print(f"New best model saved with accuracy: {best_acc:.4f}, Epoch: {epoch+1}")
            
    print(f"Training complete. Best accuracy: {best_acc:.4f}")

# Loading the model from checkpoint for additional training
def load_checkpoint(model, optimizer, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # 모델을 명시적으로 올바른 디바이스로 이동
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 옵티마이저의 상태를 올바른 디바이스로 이동
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
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
    parser.add_argument('--saving_folder', '-s', type=str, default='', help='new folder name - for diverse experiments')
    args = parser.parse_args()

    # Load dataset and split into train, validation, and test sets
    df = pd.read_csv(args.csv)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print('Dataset Loaded.')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = DeepFakeDataset(train_df, study_type=args.study_type, augment=True)
    val_dataset = DeepFakeDataset(val_df, study_type=args.study_type, augment=False)
    print('Datasets Created.')

    batch_size = 32 if args.study_type == 'face' else 16  # Adjust as needed

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    print('Train Loader Created.')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print('Validation Loader Created.')


    # num_classes, Model, Loss, Optimizer
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFakeDetectionModel(args.study_type, num_classes=num_classes)
    model.to(device)
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

    print('\nTraining Started.\n')
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs, start_epoch=start_epoch, study_type=args.study_type)