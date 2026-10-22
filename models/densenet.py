import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet(nn.Module):
    def __init__(self, emb_dims, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = densenet121(weights=weights)
        # self.model = torch.hub.load('pytorch/vision:v0.8.0', 'densenet121', pretrained=True)
        self.linear1 = nn.Linear(1000, emb_dims, bias=False)
        self.bn1 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = self.model(x)
        return F.relu(self.bn1(self.linear1(x)))
