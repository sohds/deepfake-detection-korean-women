import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import pandas as pd
import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DeepFakeDetectionModel
from dataset import DeepFakeDataset

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Testing function
def test_model(model, test_loader, study_type='face_nose'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

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

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"FPR: {fpr_rate:.4f}")
    print(f"TPR: {tpr_rate:.4f}")

    # Save ROC curve data
    with open(f'/content/drive/MyDrive/Capstone-Design/multiscaleDetect/aucPickleFiles/{args.study_type}_roc_curve_data.pkl', 'wb') as f:
        pickle.dump({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}, f)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'/content/drive/MyDrive/Capstone-Design/multiscaleDetect/rocCurve/{args.study_type}_{epoch}epoch_roc_curve.png')
    plt.close()

# Main execution with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepFake Detection Testing')
    parser.add_argument('--csv', '-f', type=str, default='/content/drive/MyDrive/Capstone-Design/multiscaleDetect/meta_data.csv', required=True, help='Path to the CSV file containing metadata for testing')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--study_type', '-t', type=str, choices=['face', 'face_nose', 'face_central', 'face_cheeks_nose'], required=True, help='Ablation study type to use')
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.csv)
    test_dataset = DeepFakeDataset(df, transform=transform)

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    model = DeepFakeDetectionModel(args.study_type)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']

    # Test the model
    test_model(model, test_loader, study_type=args.study_type)