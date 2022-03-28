import numpy as np
import torch.nn as nn
import torch
import torchvision.models
from numpy import unravel_index
from torch import Tensor
from torchvision import models
from typing import Tuple, Optional
import cv2

class RESNET_2out(nn.Module):
    def __init__(self, layer_number:int = 3, num_classes: int = 1000, color_or_grey: str = "grey") -> None:
        super(RESNET_2out, self).__init__()
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
            self.out1_1[0].in_features = self.layer1[0].conv1.out_channels#in_channels
        elif self.layer_number == 2:
            x_2out = x2
            self.out1_1[0].in_features = self.layer2[0].conv1.out_channels#in_channels
        elif self.layer_number == 3:
            x_2out = x3
            self.out1_1[0].in_features = self.layer3[0].conv1.out_channels#in_channels
        elif self.layer_number == 4:
            x_2out = x4
            self.out1_1[0].in_features = self.layer4[0].conv1.out_channels#in_channels

        y = Tensor()
        # y1 = Tensor()
        # print("x shape = ", x.shape)

        all_filters_bs = []
        # all_ind_x_bs = []
        # all_ind_y_bs = []
        y_bs = []
        # print(x.shape)
        for bs in range(x_2out.shape[0]):
            all_filters = []
            feature_map_sum = []
            for i in range(x_2out.shape[1]):
                filter1 = x_2out[bs, i, :, :]
                filter1 = filter1[:, :, np.newaxis].detach().numpy()
                all_filters.append(filter1)

            np_all_filters = np.array(all_filters)

            all_filters_bs.append(np_all_filters)
            np_all_filters_bs = np.array(all_filters_bs)

            # find index of max element
            ind = unravel_index(np.argmax(np_all_filters), np_all_filters.shape)
            ind_x = int(ind[1])
            # all_ind_x_bs.append(ind_x)
            ind_y = int(ind[2])
            # all_ind_y_bs.append(ind_y)
            # print(np_all_filters[:, ind_x, ind_y, :])

            y = torch.as_tensor(np_all_filters_bs[bs, :, ind_x, ind_y, :])
            y_bs.append(np_all_filters_bs[bs, :, ind_x, ind_y, :])

            # save feature map
            feature_map_sum = sum(ele for ele in all_filters)
            feature_map_sum = np.round(feature_map_sum, 0)
            # feature_map_sum *= 255
            # print(feature_map_sum)
            cv2.imwrite("D:\\IF\\project_bacteria_recognition\\split_2021_2022\\{}.png".format(bs), feature_map_sum)

        # y1 = self.avgpool(x)
        # y1 = torch.flatten(y1, 1)
        # y1 = self.out1_1(y1)

        y = torch.as_tensor(y_bs)
        y = torch.flatten(y, 1)
        # y = np.swapaxes(y, 0, 1)
        # y = self.out1(y)
        print(y.shape)
        print(self.out1_1[0].in_features)
        y = self.out1_1(y)
        print("y shape = ", y.shape)

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

