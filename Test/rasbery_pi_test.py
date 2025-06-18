import torch
import cv2
import numpy as np
from models.faster_rcnn_detector import get_fasterrcnn_model

model = get_fasterrcnn_model(num_classes=3, backbone_type="resnet18", weights=False)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
video_path = "/test_video/test_720P.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)  # USB veya Pi Camera için uygun id

skip_n = 2      # Kaç frame’de bir tahmin yapılacak
frame_count = 0
last_outputs = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamadı!")
        break

    frame_count += 1

    if frame_count % skip_n == 1:  # Her skip_n frame’de bir tahmin yap
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 234))
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        inputs = [img_tensor]
        with torch.no_grad():
            last_outputs = model(inputs)

    # Sonuçları çizdir (her frame, eski sonuca göre)
    if last_outputs is not None:
        boxes = last_outputs[0]['boxes'].cpu().numpy()
        scores = last_outputs[0]['scores'].cpu().numpy()
        labels = last_outputs[0]['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Camera Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
