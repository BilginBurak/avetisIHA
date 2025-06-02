import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_fasterrcnn_model(num_classes=3, pretrained_backbone=True):
    """
    Faster R-CNN modeli döndürür.

    num_classes: sınıf sayısı (background + hexagon + triangle = 3)
    pretrained_backbone: True ise backbone ImageNet ile ön-eğitimli gelir
    """
    # 1) Backbone: ResNet50 + FPN
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)

    # 2) Faster R-CNN modeli: sınıf sayısı +1 (çünkü class_id=0 → background)
    model = FasterRCNN(backbone, num_classes=num_classes)

    return model
