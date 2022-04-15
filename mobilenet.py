import numpy as np
import torch.nn as nn
import torch
from torchvision import models

class Mobilenet (nn.Module):
    def __init__(self,  num_classes: int = 1000, color_or_grey: str = "grey") -> None:
        super(Mobilenet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)

        if color_or_grey == 'grey':
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(1280, num_classes),
                    nn.Softmax(dim=1))

