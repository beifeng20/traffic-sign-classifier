"""Small VGG-style CNN for 32x32 traffic sign images."""
import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 conv + BatchNorm + ReLU, then 2x2 max-pool (halves H and W)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SignNet(nn.Module):
    def __init__(self, num_classes: int = 43, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),    # 32x32 -> 16x16
            conv_block(32, 64),   # 16x16 -> 8x8
            conv_block(64, 128),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
