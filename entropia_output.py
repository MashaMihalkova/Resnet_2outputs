import numpy as np
import torch.nn as nn
import torch
import torchvision.models
from numpy import unravel_index
from torch import Tensor
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
from math import log, e
import cv2

class Entropia_2output(nn.Module):
    def __init__(self, layer_number:int = 3, num_classes: int = 1000, color_or_grey: str = "grey") -> None:
        super(Entropia_2output, self).__init__()
        model = models.resnet18(pretrained=True)
        if color_or_grey == 'grey':
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_class)

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1))
        model1 = list(model.children())
        self.conv1 = model1[0]
        self.bn1 = model1[1]
        self.relu = model1[2]
        self.maxpool = model1[3]
        self.layer1 = model1[4]
        self.layer2 = model1[5]
        self.maxpool_1output = nn.MaxPool2d((1, 1))

        self.out1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1))
        self.size_l = 128
        self.out1_1 = nn.Sequential(
            nn.Linear(self.size_l, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1))
        # self.y = Tensor
        self.layer3 = model1[6]
        self.layer4 = model1[7]

        self.avgpool = model1[8]
        self.fc = model1[9]
        self.layer_number = layer_number


    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # See note [TorchScript super()]

        global a_bs
        size_l = self.layer1[0].conv1.in_channels

        print(size_l)

        # self.out1_1[0].in_features = self.size_l
        # self.out1_1[0]

        # img = x[0, 0].detach().numpy()
        # img = img[:, :, np.newaxis]
        # img *= 255
        #
        # cv2.imwrite("D:\\IF\\project_bacteria_recognition\\split_2021_2022\\img.png", img)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("forward")

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # print(x.shape)

        x3 = self.layer3(x2)
        x4 = self.layer4(x3)


        print("OUT = ",self.layer3[0].conv1.out_channels)
        if self.layer_number == 1:
            x_2out = x1
            self.size_l = self.layer1[0].conv1.out_channels
            self.out1_1[0] = nn.Linear(self.layer1[0].conv1.out_channels, 256)
                # .in_features = self.layer1[0].conv1.out_channels#in_channels
        elif self.layer_number == 2:

            x_2out = x2
            self.size_l = self.layer2[0].conv1.out_channels
            self.out1_1[0] = nn.Linear(self.layer2[0].conv1.out_channels, 256)
            self.out1_1[0].in_features = self.layer2[0].conv1.out_channels#in_channels
        elif self.layer_number == 3:
            x_2out = x3
            self.size_l = self.layer3[0].conv1.out_channels
            self.out1_1[0] = nn.Linear(self.layer3[0].conv1.out_channels, 512)
            self.out1_1[0].in_features = self.layer3[0].conv1.out_channels#in_channels
        elif self.layer_number == 4:
            x_2out = x4
            self.out1_1[0] = nn.Linear(self.layer4[0].conv1.out_channels, 1024)
            self.size_l = self.layer3[0].conv1.out_channels
            self.out1_1[0].in_features = self.layer4[0].conv1.out_channels#in_channels

        y = Tensor()
        # y1 = Tensor()
        # print("x shape = ", x.shape)

        all_filters_bs = []
        # all_ind_x_bs = []
        # all_ind_y_bs = []
        y_bs = []

        for bs in range(x_2out.shape[0]):
            bs_x = x_2out[bs]
            a_bs = []
            for i in range(x_2out.shape[2]):
                for j in range(x_2out.shape[3]):
                    a = x_2out[bs, :, i, j]#.unsqueeze(1)

                    a_bs.append(a)

            y_bs_ = torch.stack(a_bs)
            y_bs.append(y_bs_)
            # y_bs.append(a_bs)# torch.stack(a_bs)
            # y_bs.append(a_bs)
        y = torch.stack(y_bs)
        # y = y_bs
            # print(ind)
        # y = torch.stack(y_bs)
        # y_bs = np.array(y_bs)
        # y_bs = np.swapaxes(y_bs, 1, 2)
        # y1 = torch.as_tensor(y_bs)
        # for i in range(y.shape[0]):
        #     # y1 = torch.as_tensor(y_bs[:, i, :, :])
        #     y_ = torch.flatten(y[i, :, :], 1)
        #     y_ = self.out1_1(y_)
        #     print(y_.shape)
        y = self.out1_1(y)
            # loss = F.cross_entropy(input, target)
            # entr = self.entropy2(y_.detach().numpy())


        # y1 = self.avgpool(x)
        # y1 = torch.flatten(y1, 1)
        # y1 = self.out1_1(y1)
        # y_ = torch.as_tensor(y_bs)
        # for bs in range(x_2out.shape[0]):
        #     for i in range(x_2out.shape[1]):
        #
        #         print(y.shape)
        #         # print(self.out1_1[0].in_features)
        #         y = self.out1_1(y)
        #         print("y shape = ", y.shape)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        print('x shape = ', x.shape)

        return x, y

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x1, y1 = self._forward_impl(x)

        return x1, y1
        # def forward(self, x: Tensor) -> Tensor:
    #       return self._forward_impl(x)

    def entropy2(self, labels, base=None):
        """ Computes entropy of label distribution. """

        n_labels = len(labels)

        if n_labels <= 1:
            return 0

        value, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.

        # Compute entropy
        base = e if base is None else base
        for i in probs:
            ent -= i * log(i, base)

        return ent