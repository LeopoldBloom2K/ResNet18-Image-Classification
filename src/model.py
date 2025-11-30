import torch.nn as nn
import torchvision.models as models

def get_resnet18_model(num_classes=2, pretrained=True, freeze_backbone=True):
    """
    ResNet18 모델을 로드하고 마지막 레이어를 수정합니다.
    freeze_backbone=True일 경우, Feature Extraction 방식으로 동작합니다.
    """
    model = models.resnet18(pretrained=pretrained)

    # 특성 추출(Feature Extraction) 방식: 앞단 레이어 가중치 고정
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 마지막 분류 레이어(fc) 교체 (Cat vs Dog = 2 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model