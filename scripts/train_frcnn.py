import os
import time
import csv
import torch
import traceback
import subprocess
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from dataset.faster_rcnn_dataset import FasterRCNNDataset
from torchvision.ops import box_iou
from torch.utils.tensorboard import SummaryWriter
from models.faster_rcnn_detector import get_fasterrcnn_model


def collate_fn(batch):
    return tuple(zip(*batch))

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def compute_iou(boxA, boxB):
    xA = torch.max(boxA[0], boxB[0])
    yA = torch.max(boxA[1], boxB[1])
    xB = torch.min(boxA[2], boxB[2])
    yB = torch.min(boxA[3], boxB[3])
    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / (boxA_area + boxB_area - inter_area + 1e-6)
    return iou

def evaluate_iou_metrics(all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels, iou_thresh=0.5):
    tp, fp, fn = 0, 0, 0
    for preds, pred_labels, gts, gt_labels in zip(all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels):
        if len(preds) == 0:
            fn += len(gt_labels)
            continue
        if len(gts) == 0:
            fp += len(pred_labels)
            continue
        ious = box_iou(preds, gts)
        max_ious, max_idx = ious.max(dim=1)
        tp += (max_ious > iou_thresh).sum().item()
        fp += (max_ious <= iou_thresh).sum().item()
        fn += max(0, len(gt_labels) - (max_ious > iou_thresh).sum().item())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

def get_iou_matched_labels(all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels, iou_threshold=0.5):
    filtered_pred_labels = []
    filtered_gt_labels = []
    for preds, pred_labels, gts, gt_labels in zip(all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels):
        if len(preds) == 0 or len(gts) == 0:
            continue
        ious = box_iou(preds, gts)
        for gt_idx in range(len(gts)):
            pred_idx = ious[:, gt_idx].argmax().item()
            if ious[pred_idx, gt_idx].item() >= iou_threshold:
                filtered_pred_labels.append(pred_labels[pred_idx].item())
                filtered_gt_labels.append(gt_labels[gt_idx].item())
    return filtered_gt_labels, filtered_pred_labels


def main():
    DEVICE = torch.device("cuda" if torch.cuda  .is_available() else "cpu")
    EPOCHS = 50
    BATCH_SIZE = 8
    NUM_WORKERS = 8
    SAVE_DIR = "outputs_frcnn"
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dataset = FasterRCNNDataset("dataset/train/images", "dataset/train/labels")
    val_dataset   = FasterRCNNDataset("dataset/valid/images", "dataset/valid/labels")
    #train_dataset = Subset(train_dataset, range(200))
    #val_dataset = Subset(val_dataset, range(20))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS,pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS,pin_memory=True, collate_fn=collate_fn)

    #MobileNetV3(enhızlı)
    model = get_fasterrcnn_model(num_classes=3, backbone_type="mobilenetv3", weights=True)
    # sıfırdan eğitmek için weights FALSE

    # veya daha hafif bir ResNet:
    #model = get_fasterrcnn_model(num_classes=3, backbone_type="resnet18", weights=True)

    # Eğer eski haline dönmek istersen:
    #model = get_fasterrcnn_model(num_classes=3, backbone_type="resnet50")
    model.to(DEVICE)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    # TensorBoard log klasörünü oluştur

    log_dir = "logs/tensorboard_frcnnv3/"
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_path = os.path.join(SAVE_DIR, "checkpointv3.pth")
    if os.path.exists(checkpoint_path):
        print(f">>> Checkpoint bulundu, eğitim devam ediyor: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        val_precisions = checkpoint["val_precisions"]
        val_recalls = checkpoint["val_recalls"]
        val_f1_scores = checkpoint["val_f1_scores"]
    else:
        print(">>> Checkpoint bulunamadı, eğitim sıfırdan başlıyor.")
        start_epoch = 0
        best_f1 = 0.0
        train_losses, val_losses = [], []
        val_precisions, val_recalls, val_f1_scores = [], [], []

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    print(f">>> Training on {DEVICE} ...")
    start_time = time.time()

    # CSV için hazırla
    csv_path = os.path.join(SAVE_DIR, "metricsv3.csv")
    if start_epoch == 0:
        with open(csv_path, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_precision", "val_recall", "val_f1"])

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🔁 Epoch {epoch+1}/{EPOCHS} başlıyor...")
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)

            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        all_pred_boxes, all_pred_labels = [], []
        all_gt_boxes, all_gt_labels = [], []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                model.train()
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                total_val_loss += loss.item()
                model.eval()  # Geri döndür

                outputs = model(images)
                # Tahmin ve GT kutularını topla
                for output, target in zip(outputs, targets):
                    scores = output["scores"].cpu()
                    labels = output["labels"].cpu()
                    boxes = output["boxes"].cpu()
                    mask = scores > 0.5  # Güven eşiği

                    all_pred_boxes.append(boxes[mask])
                    all_pred_labels.append(labels[mask])
                    all_gt_boxes.append(target["boxes"].cpu())
                    all_gt_labels.append(target["labels"].cpu())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)

        precision, recall, f1 = evaluate_iou_metrics(
            all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels, iou_thresh=0.5
        )
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1_scores.append(f1)
        # --- TensorBoard logging ---
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("F1/val", f1, epoch + 1)
        writer.add_scalar("Precision/val", precision, epoch + 1)
        writer.add_scalar("Recall/val", recall, epoch + 1)

        print(f"[Epoch {epoch+1}/{EPOCHS}] 🧠 Train Loss: {avg_train_loss:.4f} | 🎯 Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")

        # En iyi model kaydı
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_modelv3.pth"))

        # Checkpoint kaydı
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_precisions": val_precisions,
            "val_recalls": val_recalls,
            "val_f1_scores": val_f1_scores
        }, os.path.join(SAVE_DIR, "checkpoint.pth"))

        # CSV güncelle
        with open(csv_path, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, precision, recall, f1])

        if early_stopping(f1):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"✅ Training completed in {total_time:.2f} seconds.")
    print(f"📈 Final F1 Score: {val_f1_scores[-1]:.4f}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curvev3.png"))

    plt.figure()
    plt.plot(val_f1_scores, label="F1 Score")
    plt.title("F1 Score Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "f1_scorev3.png"))

    print("\n📊 Classification Report (IoU > 0.5):")
    filtered_gt_labels, filtered_pred_labels = get_iou_matched_labels(
        all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels, iou_threshold=0.5
    )
    print("\n📊 Classification Report (IoU > 0.5 ile eşleşenler):")
    print(classification_report(
        filtered_gt_labels,
        filtered_pred_labels,
        target_names=["hexagon", "triangle"],
        digits=4,
        zero_division=0
    ))
    writer.close()

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n💥 CUDA OOM hatası alındı. 10 saniye sonra yeniden başlatılıyor...")
            print("sys.executable:", sys.executable)
            print("sys.argv:", sys.argv)
            torch.cuda.empty_cache()  # CUDA belleğini temizle
            time.sleep(10)
            try:
                # subprocess ile yeniden başlatma
                subprocess.run([sys.executable] + sys.argv)
                sys.exit(0)
            except Exception as exec_err:
                print(f"❌ subprocess hatası: {exec_err}")
                sys.exit(1)
        else:
            print("\n❌ Beklenmeyen bir hata oluştu:")
            traceback.print_exc()