# 라이브러리 임포트
import torch
import torch.nn as nn
from torchvision.models import xception
from timm.models.swin_transformer import SwinTransformer

# Our own model
# Feature Extractor: XceptionNet
# Feature Concat + Image Classification: Swim Transformer
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_crops=3):
        super(DeepfakeDetectionModel, self).__init__()
        
        # XceptionNet for feature extraction
        self.xception = xception(pretrained=True)
        self.xception.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Freeze XceptionNet parameters
        for param in self.xception.parameters():
            param.requires_grad = False
        
        # Swin Transformer
        self.swin = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=2048 * num_crops,  # Assuming XceptionNet outputs 2048-dim features
            num_classes=2,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

    def forward(self, x):
        # x is a list of cropped images (assuming 3 crops: full face, GradCAM area 1, GradCAM area 2)
        features = []
        for crop in x:
            feat = self.xception(crop)
            features.append(feat)
        
        # Concatenate features from all crops
        combined_features = torch.cat(features, dim=1)
        
        # Reshape for Swin Transformer input
        batch_size = combined_features.shape[0]
        combined_features = combined_features.view(batch_size, -1, 1, 1)
        
        # Pass through Swin Transformer
        output = self.swin(combined_features)
        
        return output

# 사용 예시
# model = DeepfakeDetectionModel()
# dummy_input = [torch.randn(1, 3, 224, 224) for _ in range(3)]  # 3 crops
# output = model(dummy_input)
# print(output.shape)  # Should be [1, 2] for binary classification