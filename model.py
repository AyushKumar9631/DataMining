import torch
import torch.nn as nn
import torchvision.models as models


class CrowdNetV2(nn.Module):
    """
    MobileNetV2 backbone + custom regression head.
    Predicts log1p(crowd_count); use numpy.expm1() to recover real count.
    """
    def __init__(self):
        super().__init__()
        base          = models.mobilenet_v2(weights=None)
        self.backbone = base.features          # (B, 1280, 7, 7)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.head     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.head(self.pool(self.backbone(x))).squeeze(1)


def load_model(ckpt_path: str, device: torch.device) -> CrowdNetV2:
    model = CrowdNetV2().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
