# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from Resnet_2out import RESNET_2out
from Create_dataset import *
from Train import train_model
from feature_plt import *
import cv2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # путь до данных
    PATH_TO_SAVE_WEIGHTS = 'D:\\IF\\project_bacteria_recognition\\split_2021_2022\\'

    MODEL = 'ResNet'
    batch_size = 2
    num_workers = 1
    num_class = 23
    num_epoch = 10
    color_or_grey = 'grey'  # put 'grey' or 'color'
    loss = 'CE'
    optimizer = 'Adam'
    shedule_step = 5
    log_folder = 'ResNet18_greyscale'
    device = "cpu"
    #  1150*862
    PATH_TO_TRAIN = "D:\\IF\\project_bacteria_recognition\\split_2021_2022\\train"
    PATH_TO_VAL = "D:\\IF\\project_bacteria_recognition\\split_2021_2022\\test"

    train_dataset, train_dataloader = load_images(PATH_TO_TRAIN, 'grey', batch_size, num_workers)
    val_dataset, val_dataloader = load_images(PATH_TO_VAL, 'grey', batch_size, num_workers)
    # train_dataloader, val_dataloader = create_dataset(PATH_TO_TRAIN, PATH_TO_VAL, "grey", batch_size, num_workers)

    model_2out = RESNET_2out(num_classes=num_class, color_or_grey=color_or_grey)
    model_2out = model_2out.to(device)

    if loss == 'CE':
        loss = torch.nn.CrossEntropyLoss()
    else:
        loss = torch.nn.BCELoss()
    if optimizer == 'Adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        optimizer = torch.optim.Adam(model_2out.parameters(), lr=1e-4, weight_decay=1e-5)
    # else:
    #     optimizer = torch.optim.Adagrad()

    CUDA_LAUNCH_BLOCKING = 1
    # Decay LR by a factor of 0.1 every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=shedule_step, gamma=0.1)

    modell, train_loss, val_loss, train_acc, val_acc = train_model(model_2out, train_dataloader, train_dataset, val_dataloader, loss,
                                                                   optimizer, scheduler, num_epoch,
                                                                   PATH_TO_SAVE_WEIGHTS)
    input_image = cv2.imread(
        "D:\\IF\\project_bacteria_recognition\\split_2021_2022\\test\\Acinetobacter baumannii\\1.jpeg")

    # visualize_feature_map(model_2out(input_image))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
