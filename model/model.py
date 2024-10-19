import torch
import torch.nn as nn
from timm import create_model

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
    def __init__(self, num_classes=2):
        super(SwinTransformerClassifier, self).__init__()
        self.swin = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.swin(x)

# # Full Model for each ablation study
class DeepFakeDetectionModel(nn.Module):
    def __init__(self, study_type, num_classes=2):
        super(DeepFakeDetectionModel, self).__init__()
        self.study_type = study_type
        
        if study_type == 'face':
            self.classifier = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
        else:
            self.face_extractor = XceptionFeatureExtractor()
            self.secondary_extractor = XceptionFeatureExtractor()
            
            if study_type == 'face_nose':
                input_dim = 4096  # 2048 (face) + 2048 (nose)
            elif study_type == 'face_central':
                input_dim = 4096  # 2048 (face) + 2048 (central)
            else:  # face_cheeks_nose
                input_dim = 8192  # 2048 (face) + 2048 (left_cheek) + 2048 (nose) + 2048 (right_cheek)
            
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, face, secondary=None):
        if self.study_type == 'face':
            return self.classifier(face)
        else:
            face_features = self.face_extractor(face)
            if self.study_type == 'face_nose':
                secondary_features = self.secondary_extractor(secondary)
                combined_features = torch.cat((face_features, secondary_features), dim=1)
            elif self.study_type == 'face_central':
                secondary_features = self.secondary_extractor(secondary)
                combined_features = torch.cat((face_features, secondary_features), dim=1)
            elif self.study_type == 'face_cheeks_nose':
                left_cheek, nose, right_cheek = secondary.chunk(3, dim=1)
                left_cheek_features = self.secondary_extractor(left_cheek)
                nose_features = self.secondary_extractor(nose)
                right_cheek_features = self.secondary_extractor(right_cheek)
                combined_features = torch.cat((face_features, left_cheek_features, nose_features, right_cheek_features), dim=1)
            
            return self.classifier(combined_features)