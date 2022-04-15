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
from scipy.special import softmax
import cv2


class Entropia_2output_2version(nn.Module):
    def __init__(self, layer_number: int = 3, num_classes: int = 1000, color_or_grey: str = "grey") -> None:
        super(Entropia_2output_2version, self).__init__()
        model = models.resnet18(pretrained=True)
        if color_or_grey == 'grey':
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, num_class)

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1))
        model1 = list(model.children())
        self.conv1 = model1[0]
        self.bn1 = model1[1]
        self.relu = model1[2]
        self.maxpool = model1[3]
        self.layer1 = model1[4]
        self.layer2 = model1[5]
        self.maxpool_1output = nn.MaxPool2d((1, 1))
        self.size_l = 128
        # self.linear_softmax = nn.Sequential(
        #     nn.Linear(self.size_l, num_classes),
        #     nn.Softmax(dim=2))
        self.linear_softmax = nn.Sequential(
            nn.Linear(self.size_l, num_classes),
            nn.Softmax(dim=2))
        # self.y = Tensor
        self.layer3 = model1[6]
        self.layer4 = model1[7]

        self.avgpool = model1[8]
        self.fc = model1[9]
        self.layer_number = layer_number

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # See note [TorchScript super()]

        global a_bs, x_2out
        size_l = self.layer1[0].conv1.in_channels

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # print("OUT = ",self.layer3[0].conv1.out_channels)
        if self.layer_number == 1:
            x_2out = x1
            # self.size_l = self.layer1[0].conv1.out_channels
            # self.out1_1[0] = nn.Linear(self.layer1[0].conv1.out_channels, 256)
            # .in_features = self.layer1[0].conv1.out_channels#in_channels
        elif self.layer_number == 2:

            x_2out = x2
            # self.size_l = self.layer2[0].conv1.out_channels
            # self.out1_1[0] = nn.Linear(self.layer2[0].conv1.out_channels, 256)
            # self.out1_1[0].in_features = self.layer2[0].conv1.out_channels#in_channels
        elif self.layer_number == 3:
            x_2out = x3
            # self.size_l = self.layer3[0].conv1.out_channels
            # self.out1_1[0] = nn.Linear(self.layer3[0].conv1.out_channels, 512)
            # self.out1_1[0].in_features = self.layer3[0].conv1.out_channels#in_channels
        elif self.layer_number == 4:
            x_2out = x4
            # self.out1_1[0] = nn.Linear(self.layer4[0].conv1.out_channels, 1024)
            # self.size_l = self.layer3[0].conv1.out_channels
            # self.out1_1[0].in_features = self.layer4[0].conv1.out_channels#in_channels

        y = Tensor()
        # y_bs = []
        # for bs in range(x_2out.shape[0]):
        #     bs_x = x_2out[bs]
        #     a_bs = []
        #     for i in range(x_2out.shape[2]):
        #         for j in range(x_2out.shape[3]):
        #             a = x_2out[bs, :, i, j]  # .unsqueeze(1)
        #             a_bs.append(a)
        #     y_bs_ = torch.stack(a_bs)
        #     y_bs.append(y_bs_)
        # y1 = torch.stack(y_bs)
        y = torch.reshape(x_2out, (x_2out.shape[0], x_2out.shape[1], x_2out.shape[2]*x_2out.shape[3]))
        # print('y_shape = ', y.shape)
        y = y.swapaxes(2, 1)

        print('y_shape = ', y.shape)
        y = self.linear_softmax(y)

        # koef = y.shape[2] * 0.001 + 1
        y_clone = y.clone() # проверить рекв град
        # for bs in range(y.shape[0]):
        #     for p_i in range(y.shape[2]):
        #         y_clone[bs][:, p_i] = y_clone[bs][:, p_i] + 0.001
        #         y_clone[bs][:, p_i] = y_clone[bs][:, p_i] / koef
        # H = torch.stack([(y_clone[bs][:, :] * y_clone[bs][:, :].log()).sum(dim=1) for bs in
        #                  range(y_clone.shape[0])])
        # H_min = torch.argmin(H, dim=1)
        H_min = [0,0]
        y_out = torch.stack([y_clone[bs][H_min[bs]] for bs in range(y_clone.shape[0])])

        # y = self.fc(y)
        # SOFTMAX
        # m = nn.Softmax(dim=2)
        # y1 = m(y)
        # y1 = self.softmax(y)
        # y = torch.nn.Softmax(dim=2)(y)
        # y = softmax(y)

        # y = self.softmax(y)

        #  shape = (bs, 128*128, 23)
        #  (bs, 128*128, p)
        # p+=0.001 after p/=1.023

        # koef = y.shape[2] * 0.001 + 1
        # for bs in range(y.shape[0]):
        #     for p_i in range(y.shape[2]):
        #         y[bs][:, p_i] += 0.001
        #         y[bs][:, p_i] /= koef
        #
        # # H = (p * p.log()).sum(dim = 2)
        #
        # H = torch.stack([(y[bs][:, :] * y[bs][:, :].log()).sum(dim=1) for bs in range(y.shape[0])])
        # # h = []
        # # for bs in range(y.shape[0]):
        # #     h.append((y[bs][:, :] * y[bs][:, :].log()).sum(dim=1))
        # # H = torch.stack(h)
        #
        # H_min = torch.argmin(H, dim=1)
        #
        #
        # # y_ = []
        # y_out = torch.stack([y[bs][H_min[bs]] for bs in range(y.shape[0])])

        # for bs in range(y.shape[0]):
        #     y_.append(y[bs][H_min[bs]])
        # y_out = torch.stack(y_)



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

        # print('x shape = ', x.shape)

        return x, y_out

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
