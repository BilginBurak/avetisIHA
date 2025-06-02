
import os
import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset.faster_rcnn_dataset import FasterRCNNDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Testing on device: {DEVICE}")

def load_model(model_path, num_classes=3):
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def run_test(model, test_img_dir, output_dir, conf_thresh=0.5):
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    class_names = ["background", "hexagon", "triangle"]

    for img_file in image_files:
        img_path = os.path.join(test_img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Image could not be read: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.tensor(np.transpose(img_rgb, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{class_names[label]}: {score:.2f}"
            cv2.putText(img, text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        save_path = os.path.join(output_dir, f"pred_{img_file}")
        cv2.imwrite(save_path, img)
        print(f"✔ Saved: {save_path}")

if __name__ == "__main__":
    model_path = "outputs_frcnn/best_model.pth"
    test_image_dir = "dataset/test/images"
    save_dir = "outputs_frcnn/predictions"

    model = load_model(model_path)
    run_test(model, test_image_dir, save_dir)
