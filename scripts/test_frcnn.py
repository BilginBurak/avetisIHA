import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

from dataset.faster_rcnn_dataset import FasterRCNNDataset
from models.faster_rcnn_detector import get_fasterrcnn_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3  # background + hexagon + triangle

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = interArea / float(box1Area + box2Area - interArea + 1e-8)
    return iou

def evaluate_model(model, dataloader, iou_thresh=0.5, conf_thresh=0.5, num_classes=NUM_CLASSES, output_dir="outputs_frcnn/test_frcnn"):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_gt_labels = []
    all_pred_labels = []
    all_scores = []
    all_gts_bin = []
    all_preds_bin = []

    for images, targets in dataloader:
        images = [img.to(DEVICE) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()

            mask = pred_scores >= conf_thresh
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]

            matched_gt = set()
            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                ious = [compute_iou(pb, gb) for gb in gt_boxes] if len(gt_boxes) else []
                max_iou = max(ious) if ious else 0
                if ious and max_iou >= iou_thresh:
                    idx = np.argmax(ious)
                    if idx not in matched_gt:
                        matched_gt.add(idx)
                        all_pred_labels.append(pl)
                        all_gt_labels.append(gt_labels[idx])
                        all_scores.append(ps)
                        all_gts_bin.append(1)
                        all_preds_bin.append(1)
                    else:
                        all_pred_labels.append(pl)
                        all_gt_labels.append(0)
                        all_scores.append(ps)
                        all_gts_bin.append(0)
                        all_preds_bin.append(1)
                else:
                    all_pred_labels.append(pl)
                    all_gt_labels.append(0)
                    all_scores.append(ps)
                    all_gts_bin.append(0)
                    all_preds_bin.append(1)
            unmatched = set(range(len(gt_labels))) - matched_gt
            for idx in unmatched:
                all_pred_labels.append(0)
                all_gt_labels.append(gt_labels[idx])
                all_scores.append(0)
                all_gts_bin.append(1)
                all_preds_bin.append(0)

    # Confusion Matrix
    if all_pred_labels and all_gt_labels:
        cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=[1,2])
        print("Confusion Matrix:\n", cm)
        np.savetxt(os.path.join(output_dir, "confusion_matrix.txt"), cm, fmt="%d")

        plt.figure(figsize=(5,5))
        plt.imshow(cm, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0,1], ["hexagon", "triangle"])
        plt.yticks([0,1], ["hexagon", "triangle"])
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

    # Precision-Recall ve AP
    if all_gts_bin and all_scores:
        precision, recall, _ = precision_recall_curve(all_gts_bin, all_scores)
        ap = average_precision_score(all_gts_bin, all_scores)
        print(f"Average Precision (AP): {ap:.4f}")

        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(f"Average Precision (AP): {ap:.4f}\n")

        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()

    # Sınıf bazlı rapor
    report = classification_report(
        all_gt_labels, all_pred_labels,
        labels=[1,2],
        target_names=["hexagon", "triangle"]
    )
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    test_dataset = FasterRCNNDataset("dataset/test/images", "dataset/test/labels")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_fasterrcnn_model(num_classes=NUM_CLASSES, backbone_type="resnet18", weights=False)
    model.load_state_dict(torch.load("outputs_frcnn/best_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    evaluate_model(model, test_loader)
