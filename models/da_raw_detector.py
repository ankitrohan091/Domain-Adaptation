# models/da_raw_detector.py
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from models.cbam import CBAM
from models.utils import GradientReversalLayer

class DA_RAW_FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(DA_RAW_FasterRCNN, self).__init__()

        # Backbone with FPN
        self.backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT)
        self.detector = FasterRCNN(self.backbone, num_classes=num_classes)

        # CBAM attention on P4 and P5 (style alignment)
        self.cbam_p4 = CBAM(256)
        self.cbam_p5 = CBAM(256)

        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Image-level domain classifier
        self.img_domain_cls = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

        # Instance-level projection head for contrastive learning
        self.instance_proj = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, images, targets=None):
        if self.training:
            return self.detector(images, targets)
        else:
            return self.detector(images)

    def extract_fpn_features(self, x):
        # Output from backbone for FPN levels
        return self.backbone.body(x)

    def image_level_alignment(self, fpn_outs):
        # Apply CBAM to P4 and P5
        p4 = self.cbam_p4(fpn_outs['2'])
        p5 = self.cbam_p5(fpn_outs['3'])
        p4 = self.grl(p4)
        p5 = self.grl(p5)
        # Domain classification logits
        dom_logits_p4 = self.img_domain_cls(p4)
        dom_logits_p5 = self.img_domain_cls(p5)
        return dom_logits_p4, dom_logits_p5

    def project_instance_features(self, roi_features):
        return self.instance_proj(roi_features)
