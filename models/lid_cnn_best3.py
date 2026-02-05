import torch
import torch.nn as nn
import torch.nn.functional as F


class LIDCNN(nn.Module):
    """
    Language Identification CNN
    Input: (B, 1, n_mels=128, T)
    Output: (B, num_languages)
    """

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # -------- Block 1 --------
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 → 64

            # -------- Block 2 --------
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 → 32

            # -------- Block 3 --------
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 → 16
        )

        # Adaptive pooling handles variable time dimension
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: (B, 1, 128, T)
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
