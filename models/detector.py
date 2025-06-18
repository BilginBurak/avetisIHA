import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Özellik çıkarım katmanları
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # -> 208x208
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 104x104
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 52x52
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 26x26
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 128, 1, 1]
        )

        # Tam bağlantılı katmanlar
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)

        # Çıkış katmanları
        self.cls_head = nn.Linear(64, num_classes)  # sınıf çıktısı
        self.box_head = nn.Linear(64, 4)            # bbox çıktısı: [xc, yc, w, h]

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        class_logits = self.cls_head(x)
        bbox_preds = self.box_head(x)
        return class_logits, bbox_preds
