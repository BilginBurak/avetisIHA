import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class FasterRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        label_path = os.path.join(self.label_dir, img_filename.replace(".jpg", ".txt")
                                                  .replace(".png", ".txt")
                                                  .replace(".jpeg", ".txt"))

        #TARGET_W, TARGET_H = 320, 180

        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Yeniden boyutlandır
        #img = cv2.resize(img, (TARGET_W, TARGET_H))
        h, w = img.shape[:2]

        # Normalize et [0, 1] ve [H,W,C] → [C,H,W]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts)
                    # normalize (YOLO)   → pixel koordinatlarına çevir
                    cx *= w
                    cy *= h
                    bw *= w
                    bh *= h
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)  # 0 → background olduğu için +1

        # Boş ise shape (0, 4) olmalı
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target
