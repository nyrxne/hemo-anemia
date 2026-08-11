"""
H.E.M.A. — Model definition.

Baseline architecture: EfficientNet-B0, ImageNet-pretrained, with the classifier
head replaced for binary anemia/non-anemic classification. A two-phase fine-tuning
strategy is used, standard practice for small medical-imaging datasets (~250 train
images here — too few to fine-tune a full CNN from the first epoch without
overfitting or destroying the pretrained features):

  Phase 1 (head-only):   freeze the backbone, train only the new classifier head.
  Phase 2 (fine-tune):   unfreeze the backbone, continue training end-to-end at a
                          lower learning rate.
"""
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(architecture: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = 2):
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        backbone_params = net.features.parameters()
    elif architecture == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        backbone_params = net.features.parameters()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return net, backbone_params


def set_backbone_trainable(backbone_params, trainable: bool):
    for p in backbone_params:
        p.requires_grad = trainable
