import torch.nn as nn
from torchvision import models

class MaskModel(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        # use pretrained model
        self.network = models.resnet34(pretrained=pretrained)
        # Replace Last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)