# dataset/detection_dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def letterbox(image, new_shape=(416, 416), color=(114, 114, 114)):
    shape = image.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, int(dh), int(dh), int(dw), int(dw),
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, dw, dh

class DetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=416, transform=None):
        self.image_dir  = image_dir
        self.label_dir  = label_dir
        self.img_size   = img_size
        self.transform  = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name  = self.image_files[idx]
        img_path  = os.path.join(self.image_dir, img_name)
        label_path= os.path.join(self.label_dir, img_name.rsplit(".", 1)[0] + ".txt")

        # Dosyayı oku (Unicode‐safe)
        img = imread_unicode(img_path)
        if img is None:
            print(f"⚠️ Skipped corrupted/unreadable image: {img_path}")
            # Bu sample'ı atlamak için, kendimize göre bir “dummy” dönebiliriz:
            # (örneğin sıfır görüntü ve empty label)
            dummy_img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
            dummy_label = torch.zeros((0,5), dtype=torch.float32)
            return torch.tensor(dummy_img), dummy_label

        # Letterbox işlemi
        img, r, dw, dh = letterbox(img, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Etiketleri oku
        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls, cx, cy, w, h = map(float, line.split())
                    cx *= r * self.img_size
                    cy *= r * self.img_size
                    w  *= r * self.img_size
                    h  *= r * self.img_size
                    x1 = cx - w / 2 + dw
                    y1 = cy - h / 2 + dh
                    x2 = cx + w / 2 + dw
                    y2 = cy + h / 2 + dh
                    cls = int(cls)
                    targets.append([cls, x1, y1, x2, y2])
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        return img_tensor, targets_tensor

def custom_collate_fn(batch):
    images  = []
    targets = []
    for img, lbl in batch:
        images.append(img)
        targets.append(lbl)
    images = torch.stack(images, dim=0)
    return images, targets
