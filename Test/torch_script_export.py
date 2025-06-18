import torch
from models.faster_rcnn_detector import get_fasterrcnn_model

model = get_fasterrcnn_model(num_classes=3, backbone_type="resnet18", weights=False)
model.load_state_dict(torch.load("../outputs_frcnn/best_model.pth", map_location="cpu"))
model.eval()

# Sadece model üzerinde script uygula
scripted_model = torch.jit.script(model)
scripted_model.save("../outputs_frcnn/best_model_scripted.pt")

print("TorchScript script() export tamam!")
