import torch
import cv2
import numpy as np
import os
import sys

# Kullanıcıdan model, video ve backbone seçimi iste
MODEL_PATH = "best_model_scripted.pt"
VIDEO_PATH = "/test_video/test_720P.mp4"  # Dosya yolunu tam olarak gir
USE_CAMERA = False  # True yaparsan webcam ile çalışır


# Frame atlama (ör: 2 = her 2 karede bir tahmin)
SKIP_N = 4
INPUT_WIDTH, INPUT_HEIGHT = 416, 234

# Modeli yükle
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# Video veya kamera aç
if USE_CAMERA:
    cap = cv2.VideoCapture(0)
    print("Kamera ile başlatıldı.")
else:
    if not os.path.isfile(VIDEO_PATH):
        print(f"Video dosyası bulunamadı: {VIDEO_PATH}")
        sys.exit(1)
    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"Video ile başlatıldı: {VIDEO_PATH}")

frame_count = 0
last_outputs = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video veya kamera sonlandı/okunamıyor.")
        break

    frame_count += 1

    # Her SKIP_N karede bir inference yap
    if frame_count % SKIP_N == 1:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        inputs = [img_tensor]
        with torch.no_grad():
            last_outputs = model(inputs)

    # Sonuçları çizdir (eski veya yeni)
    if last_outputs is not None:
        boxes = last_outputs[0].get('boxes', []).cpu().numpy()
        scores = last_outputs[0].get('scores', []).cpu().numpy()
        labels = last_outputs[0].get('labels', []).cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # FPS'yi görsel olarak göstermek için (isteğe bağlı)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
