import numpy as np
import torch.nn as nn
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import models
from Create_dataset import *
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch
import torchvision.models
from numpy import unravel_index
from torch import Tensor
from torchvision import models
from typing import Tuple, Optional
import cv2
import pickle
# Источник: https://pythonim.ru/moduli/pickle-python?
'''
    About implementation,
    you just have to train a neural network,
    then select one of the layers
    (usually the ones right before the fully connected layers or the first fully connected one),
     run the neural network on your dataset,
      store all the feature vectors,
    then train an SVM with a different library (e.g sklearn).
'''
batch_size = 1
num_workers = 1
num_class = 23
num_epochs = 1
color_or_grey = 'grey'  # put 'grey' or 'color'
loss = 'CE'
optimizer = 'Adam'
shedule_step = 5
log_folder = 'ResNet18_greyscale'
device = "cpu"
PATH_TO_TRAIN = "D:\\Projects\\bacteria_recognitions\\datasets\\2021_2022\\our_data\\train_val_sbalans\\train"
PATH_TO_VAL = "D:\\Projects\\bacteria_recognitions\\datasets\\2021_2022\\our_data\\train_val_sbalans\\val"
PATH_TO_TEST = "D:\\Projects\\bacteria_recognitions\\datasets\\2021_2022\\our_data\\train_val_sbalans\\test"

# train_dataloader, val_dataloader, test_dataloader = create_dataset(PATH_TO_TRAIN, PATH_TO_VAL, PATH_TO_TEST, "grey",
#                                                                    batch_size, num_workers)

# print(train_dataloader.dataset.samples)
train_dataset, train_dataloader = load_images(PATH_TO_TRAIN, 'grey', batch_size, num_workers)
# val_dataset, val_dataloader = load_images(PATH_TO_VAL, 'grey', batch_size, num_workers)
test_dataset, test_dataloader = load_images(PATH_TO_TEST, 'grey', batch_size, num_workers)

# загрузка предобученной сети
# PATH = 'D:\\IF\\project_bacteria_recognition\\cnn+svm\\3'
PATH = 'D:\\Projects\\bacteria_recognitions\\saved_weights\\Exp_Resnet_total\\Resnet_RM_dropout\\seed10_Resnet_input_b23_wd5_q0_u1-size_1024\\3'
num_class = 23
model = models.resnet18(pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_class)
# model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
model.to('cpu')
list(model.modules())
my_model = nn.Sequential(*list(model.modules())[:-1])
# сохранение всех признаков по всем картинкам из обуч выборке

class My_model(nn.Module):
    def __init__(self, layer_number:int = 3, num_classes: int = 1000, color_or_grey: str = "grey", PATH: str='') -> None:
        super(My_model, self).__init__()
        model = models.resnet18(pretrained=True)
        if color_or_grey == 'grey':
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_class)
        # model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        model.eval()
        model1 = list(model.children())
        self.conv1 = model1[0]
        self.bn1 = model1[1]
        self.relu = model1[2]
        self.maxpool = model1[3]
        self.layer1 = model1[4]
        self.layer2 = model1[5]
        self.layer3 = model1[6]
        self.layer4 = model1[7]
        self.avgpool = model1[8]



    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x4 = self.avgpool(x4)
        return x4

    def extr_featueres(self, x: Tensor) -> Tensor:
        x1 = self._forward_impl(x)
        return x1


if __name__ == '__main__':
    my_model = My_model(3, num_class, 'grey', PATH)
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            dataloader = train_dataloader
            my_model.eval()
        features_all = []
        labels_all = []
        for inputs, labels, path in tqdm(dataloader):
                features = my_model.extr_featueres(inputs)
                # torch.save(features, 'features.txt')
                # torch.save(labels, 'labels.txt')
                # with open('features.txt', 'a+') as file:
                    # file.write(np.array(features.detach()))
                    # k = np.array(features.detach())
                    # np.save(file, np.array(features.detach()))
                # with open('labels.txt', 'a+') as file1:
                    # file1.write(np.array(labels.detach()))
                    # np.save(file1, np.array(labels.detach()))
                # file = open('features', 'a+')
                # pickle.dump(features, file)
                # file.write(features)
                # file.close()

                # features_all.append(features)
                # file1 = open('labels', 'a+')
                # pickle.dump(labels, file1)
                # file.write(labels)
                # file.close()

                # labels_all.append(labels)
        # print(features_all)
        features_all_tensor = torch.stack(features_all)
        features_all_tensor = features_all_tensor.reshape((features_all_tensor.shape[0]*features_all_tensor.shape[1], 512)).detach().cpu().numpy()
        labels_tensor = torch.stack(labels_all)
        labels_tensor = labels_tensor.reshape((labels_tensor.shape[0]*labels_tensor.shape[1]))

        # nsamples, nx, ny, nz = features_all_tensor.shape
        # X = features_all_tensor.reshape((nsamples, nx * ny * nz))
        nsamples, nx = features_all_tensor.shape
        X = features_all_tensor.reshape((nsamples, nx))

        svc = SVC()
        model_svm = svc.fit(X, labels_tensor.numpy())

        labels_all = []
        features_all_test = []
        for inputs, labels, path in tqdm(test_dataloader):
                features_test = my_model.extr_featueres(inputs)
                features_all_test.append(features_test)
                labels_all.append(labels)
        # print(features_all)
        features_all_test_tensor = torch.stack(features_all_test).detach().cpu().numpy()
        features_all_test_tensor = features_all_test_tensor.reshape((features_all_tensor.shape[0]*features_all_tensor.shape[1], 512))
        labels_tensor = torch.stack(labels_all)
        labels_tensor = labels_tensor.reshape((labels_tensor.shape[0] * labels_tensor.shape[1]))

        nsamples, nx = features_all_test_tensor.shape
        X = features_all_test_tensor.reshape((nsamples, nx))

        y_pred_svc = model_svm.predict(X)
        # print("\n Done")
        print("\n Accuracy of SVM: ", accuracy_score(labels_tensor.numpy(), y_pred_svc))

# 7832

# y_train - labels
# X-train - feature vectors
#
# svc = SVC()
# model = svc.fit(X_train,y_train)
# y_pred_svc = model.predict(X_test)
# print("\n Done")
# print("\n Accuracy of SVM: ",accuracy(y_test,y_pred_svc))

