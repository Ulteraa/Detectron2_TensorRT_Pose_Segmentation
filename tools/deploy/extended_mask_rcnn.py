import torch
import torch.nn as nn
from detectron2.modeling import build_model
from detectron2.structures import ImageList


# Custom model class extending Mask R-CNN
class ExtendedMaskRCNN(nn.Module):
    def __init__(self, cfg):
        super(ExtendedMaskRCNN, self).__init__()

        # Build Mask R-CNN backbone
        self.mask_rcnn = build_model(cfg)

        # New head for occlusion estimation
        in_features = self.mask_rcnn.backbone.out_channels
        self.occlusion_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output a value between 0 and 1
        )

    def forward(self, images, targets=None):
        # Forward pass through Mask R-CNN
        if self.training:
            losses = self.mask_rcnn(images, targets)
        else:
            predictions = self.mask_rcnn(images)
            losses = {}

        # Forward pass through occlusion head
        features = self.mask_rcnn.backbone(images.tensor)
        occlusion_scores = self.occlusion_head(features)

        return {"losses": losses, "occlusion_scores": occlusion_scores}


# Example usage
cfg_path = "path/to/your/config.yaml"
cfg = torch.load(cfg_path)
model = ExtendedMaskRCNN(cfg)

# Example input data (image batch)
image_batch = torch.randn(2, 3, 256, 256)  # Batch of 2 images
image_list = ImageList(image_batch, [(256, 256)] * 2)
output = model(image_list)

# Access instance segmentation predictions and occlusion scores
instance_predictions = output["instances"]
occlusion_scores = output["occlusion_scores"]

print(instance_predictions)
print(occlusion_scores)
