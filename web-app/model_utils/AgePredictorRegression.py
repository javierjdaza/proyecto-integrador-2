import torch.nn as nn
from torchvision import transforms as T, models


class EfficientNetV2Regression(nn.Module):
    def __init__(self):
        super().__init__()
        # Congelar lo mismo
        self.backbone = models.efficientnet_v2_l(weights='IMAGENET1K_V1')

        # Congelar todo excepto últimas 2 capas (features.6 y features.7)
        for name, param in self.backbone.named_parameters():
            if "features.6" not in name and "features.7" not in name:
                param.requires_grad = False
        
        # Obtener dimensión de features de entrada
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),     
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),         
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),     
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),        
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),        
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)
