import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def test_max_batch_size(img_size=(3, 416, 416), max_try=16):
    print(f"🧪 Test: {img_size} için maksimum batch_size ölçülüyor...")
    for bs in range(1, max_try + 1):
        try:
            inputs = torch.randn(bs, *img_size).cuda()
            model = fasterrcnn_resnet50_fpn(num_classes=3).cuda()
            model.eval()  # 🔧 eval moduna geç!
            with torch.no_grad():
                model(inputs)
            print(f"✅ Başarılı batch_size: {bs}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM! Maksimum güvenli batch_size ≈ {bs - 1}")
                torch.cuda.empty_cache()
                break
            else:
                raise e

test_max_batch_size()
