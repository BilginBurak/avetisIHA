import torch
import cv2
import numpy as np
from models.faster_rcnn_detector import get_fasterrcnn_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODELİ YÜKLE
model = get_fasterrcnn_model(num_classes=3, backbone_type="mobilenetv3", weights=False)
model.load_state_dict(torch.load("outputs_frcnn/best_modelv3.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

class_names = ["background", "hexagon", "triangle"]

def predict_and_draw(frame, model, conf_thresh=0.5):
    # BGR -> RGB, [0,255] -> [0,1]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)[0]
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame

def process_video(video_path, output_path, conf_thresh=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = predict_and_draw(frame, model, conf_thresh=conf_thresh)
        if out:
            out.write(result_frame)
        cv2.imshow('Prediction', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "Test/test_video/Test_720P.mp4"  # <<< BURAYA VİDEO YOLUNU GİR
    output_path = "outputs_frcnn/test_frcnnv3/prediction_outputv3.mp4"  # Kaydedilmiş video istersen
    process_video(video_path, output_path, conf_thresh=0.5)
