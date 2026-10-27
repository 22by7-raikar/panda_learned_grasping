"""Multi-grasp detector of Chu, Xu & Vela, "Real-World Multiobject, Multigrasp Detection"
(IEEE RA-L 2018), built from torchvision parts.

    RGD image -> ResNet-50 conv1..res4 (stride 16, 1024 ch)
              -> Grasp Proposal Network (RPN: graspable vs not, plus box deltas)
              -> RoIAlign 14x14 -> res5 + 7x7 avgpool (2048)
              -> fc 20: orientation class (19 angle bins + "no grasp")
              -> fc 80: unrotated box (x, y, w, h) per class

Why this shape: the proposal stage only answers "is there a graspable region here",
which does not depend on orientation. The second stage looks at each region's features
and classifies the angle, with per-class NMS keeping several grasps at different angles
on the same object.
"""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from grasp_learning.rectangle import NUM_ANGLE_BINS, GraspRect

# The RGD input reuses ImageNet statistics, as in the paper: depth replaces blue.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


class ResNetC4(nn.Module):
    """ResNet-50 up to and including res4 (layer3): stride 16, 1024 channels."""

    def __init__(self, trunk):
        super().__init__()
        self.body = nn.Sequential(OrderedDict([
            ("conv1", trunk.conv1), ("bn1", trunk.bn1), ("relu", trunk.relu), ("maxpool", trunk.maxpool),
            ("layer1", trunk.layer1), ("layer2", trunk.layer2), ("layer3", trunk.layer3),
        ]))
        self.out_channels = 1024

    def forward(self, x):
        return self.body(x)


class Res5Head(nn.Module):
    """res5 (layer4) on each 14x14 RoI, then global average pooling: 2048-d features."""

    def __init__(self, trunk):
        super().__init__()
        self.layer4 = trunk.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.pool(self.layer4(x)), 1)


def build_model(pretrained: bool = True, image_size: int = 320, num_angle_bins: int = NUM_ANGLE_BINS,
                box_score_thresh: float = 0.3, detections_per_img: int = 50) -> FasterRCNN:
    trunk = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    # Freeze the stem and res2 like the paper's Faster R-CNN setup; BN stays frozen
    # because batches of a few images give poor batch statistics.
    for module in (trunk.conv1, trunk.bn1, trunk.layer1):
        for p in module.parameters():
            p.requires_grad = False
    for m in trunk.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    anchors = AnchorGenerator(sizes=((16, 32, 64, 128),), aspect_ratios=((0.33, 0.5, 1.0, 2.0, 3.0),))
    return FasterRCNN(
        backbone=ResNetC4(trunk),
        num_classes=None,  # given by the predictor
        min_size=image_size,
        max_size=image_size,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        rpn_anchor_generator=anchors,
        box_roi_pool=MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2),
        box_head=Res5Head(trunk),
        box_predictor=FastRCNNPredictor(2048, num_angle_bins + 1),
        box_score_thresh=box_score_thresh,
        box_detections_per_img=detections_per_img,
    )


def freeze_batchnorm(model: nn.Module):
    """Call after model.train(): keep BatchNorm layers in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def make_rgd(rgb, depth, depth_range=None, depth_channel=2):
    """RGB (H, W, 3 uint8) + depth (H, W metres) -> float tensor (3, H, W) in [0, 1].

    Depth is min-max normalised over the valid pixels (or depth_range) and replaces one
    colour channel: blue (2) in Chu et al.; the lab's benchmarking paper describes its
    ResNet variant as replacing green (1). Train and deploy with the same choice.
    """
    rgb = torch.as_tensor(rgb).float() / 255.0
    d = torch.as_tensor(depth).float()
    valid = torch.isfinite(d) & (d > 0)
    if depth_range is None:
        lo, hi = (d[valid].min(), d[valid].max()) if valid.any() else (torch.tensor(0.0), torch.tensor(1.0))
    else:
        lo, hi = (torch.tensor(float(v)) for v in depth_range)
    d = torch.where(valid, d, hi)  # holes look like background
    d = ((d - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
    channels = [rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    channels[depth_channel] = d
    return torch.stack(channels, dim=0)


@torch.no_grad()
def predict(model: FasterRCNN, rgd: torch.Tensor, top_k: int = 10, min_score: float = 0.3):
    """Run the detector on one RGD tensor and return GraspRects, best first."""
    model.eval()
    out = model([rgd])[0]
    rects = [GraspRect.from_detection(b.tolist(), int(l), float(s))
             for b, l, s in zip(out["boxes"], out["labels"], out["scores"]) if float(s) >= min_score]
    rects.sort(key=lambda r: -r.score)
    return rects[:top_k]
