import torch
from models.faster_rcnn_detector import get_fasterrcnn_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli tanımla ve yükle
model = get_fasterrcnn_model(num_classes=3, backbone_type="mobilenetv3", weights=False)
model.load_state_dict(torch.load(r"C:\avetisIHA\outputs_frcnn\best_modelv3.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ONNX için dummy input (tek görüntü, batch boyutlu)
dummy_input = torch.randn(1, 3, 416, 416).to(DEVICE)

# Export et (aynı klasöre kaydedilir)
torch.onnx.export(model,
                  dummy_input,
                  "C:/avetisIHA/outputs_frcnn/frcnn_mobilenetv3.onnx",
                  input_names=["input"],
                  output_names=["boxes", "labels", "scores"],
                  opset_version=11,
                  dynamic_axes={"input": {0: "batch_size"},
                                "boxes": {0: "num_detections"},
                                "labels": {0: "num_detections"},
                                "scores": {0: "num_detections"}})
