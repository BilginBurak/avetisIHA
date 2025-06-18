import torchvision
from torchvision.models.detection import FasterRCNN, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_fasterrcnn_model(num_classes=3, backbone_type="mobilenetv3", weights=True):
    """
    Hızlı backbone seçimi ile Faster R-CNN modeli döndürür.
    backbone_type: 'resnet50', 'resnet18', 'mobilenetv3'
    num_classes: sınıf sayısı (background + hexagon + triangle = 3)
    pretrained_backbone: True ise backbone ImageNet ile ön-eğitimli gelir
    """
    if backbone_type == "resnet50":
        backbone = resnet_fpn_backbone('resnet50', weights="DEFAULT" if weights else None)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif backbone_type == "resnet18":
        backbone = resnet_fpn_backbone('resnet18', weights="DEFAULT" if weights else None)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif backbone_type == "mobilenetv3":
        weights_enum = (
            torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            if weights else None
        )
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights_enum)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                   num_classes)
    else:
        raise ValueError("backbone_type 'resnet50', 'resnet18', 'mobilenetv3' olmalı")
    return model