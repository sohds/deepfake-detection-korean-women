import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DeepFakeDetectionModel
from dataset import DeepFakeDataset
from sklearn.model_selection import train_test_split
import os

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# Testing function
def test_model(model, test_loader, study_type='face_nose'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []
    
    roc_dir = os.path.join(args.output_roc, args.saving_folder)
    pkl_dir = os.path.join(args.output_pkl, args.saving_folder)
    createDirectory(roc_dir)
    createDirectory(pkl_dir)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing [Test Set]"):
            face = batch['face'].to(device)

            if study_type == 'face':
                secondary = None
            elif study_type == 'face_nose':
                secondary = batch['nose'].to(device)
            elif study_type == 'face_central':
                secondary = batch['central'].to(device)
            else:  # face_cheeks_nose
                secondary = torch.cat((batch['left_cheek'], batch['nose'], batch['right_cheek']), dim=1).to(device)

            labels = batch['label'].to(device)
            if study_type == 'face':
                outputs = model(face)
            else:
                outputs = model(face, secondary)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1 (deepfake)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr_rate = fp / (fp + tn)
    tpr_rate = tp / (tp + fn)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"FPR: {fpr_rate:.4f}")
    print(f"TPR: {tpr_rate:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save ROC curve data
    with open(f'{pkl_dir}/{args.study_type}_roc_curve_data.pkl', 'wb') as f:
        pickle.dump({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}, f)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{roc_dir}/{args.study_type}_{epoch}epoch_roc_curve.png')
    plt.close()

# Main execution with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepFake Detection Testing')
    parser.add_argument('--csv', '-f', type=str, default='/content/drive/MyDrive/Capstone-Design/multiscaleDetect/meta_data.csv', required=True, help='Path to the CSV file containing metadata for testing')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--study_type', '-t', type=str, choices=['face', 'face_nose', 'face_central', 'face_cheeks_nose'], required=True, help='Ablation study type to use')
    parser.add_argument('--output_roc', '-o', type=str, default='/content/drive/MyDrive/Capstone-Design/multiscaleDetect/rocCurve', help='Directory to save roc curve plots')
    parser.add_argument('--output_pkl', '-p', type=str, default='/content/drive/MyDrive/Capstone-Design/multiscaleDetect/aucPickleFiles', help='Directory to save roc curve pkl files')
    parser.add_argument('--saving_folder', '-s', type=str, default='', help='new folder name - for diverse experiments')
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.csv)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_dataset = DeepFakeDataset(test_df, study_type=args.study_type, augment=False)
    print('Test Dataset Loaded.')
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    # Create DataLoader
    batch_size = 32 if args.study_type == 'face' else 16  # Adjust as needed
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print('Test DataLoader Created.')

    # Load model
    num_classes = 2
    model = DeepFakeDetectionModel(args.study_type, num_classes = num_classes)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded model from epoch {epoch}")
    
    print(f"Start testing model with study type: {args.study_type}")
    # Test the model
    test_model(model, test_loader, study_type=args.study_type)