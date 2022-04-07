# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from Resnet_2out import RESNET_2out
from Create_dataset import *
from Train import train_model
from feature_plt import *
from entropia_output import *
from ResNet_2output_tensor import *
import cv2
from tqdm import tqdm


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
    PATH_TO_TRAIN = "D:\\IF\\project_bacteria_recognition\\split_2021_2022\\train_val_sbalans\\train"
    PATH_TO_VAL = "D:\\IF\\project_bacteria_recognition\\split_2021_2022\\train_val_sbalans\\val"
    PATH_TO_TEST = "D:\\IF\\project_bacteria_recognition\\split_2021_2022\\test"

    # train_dataloader, val_dataloader, test_dataloader = create_dataset(PATH_TO_TRAIN, PATH_TO_VAL, PATH_TO_TEST, "grey",
    #                                                                    batch_size, num_workers)

    # print(train_dataloader.dataset.samples)
    train_dataset, train_dataloader = load_images(PATH_TO_TRAIN, 'grey', batch_size, num_workers)
    val_dataset, val_dataloader = load_images(PATH_TO_VAL, 'grey', batch_size, num_workers)
    test_dataset, test_dataloader = load_images(PATH_TO_VAL, 'grey', batch_size, num_workers)



    # model_2out = RESNET_2out(layer_number = 2,num_classes=num_class, color_or_grey=color_or_grey)
    # model_2out = model_2out.to(device)

    # model_2out = RESNET_2out_tensor(layer_number=4, num_classes=num_class, color_or_grey=color_or_grey)
    # model_2out = model_2out.to(device)

    model_entrop = Entropia_2output(layer_number=2, num_classes=num_class, color_or_grey=color_or_grey)
    model_2out = model_entrop.to(device)

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
    # train_loss = [[222.9, 9929, 0, 8989, 88.12, 9082,43],[22,466,332,634,3242,8,9]]
    #
    # with open('D:\\IF\\project_bacteria_recognition\\split_2021_2022\\text.txt', 'a+') as f:
    #     f.write('train_loss: \n')
    #     for listitem in train_loss:
    #         f.write('%s\n' % listitem)



    modell, train_loss, val_loss, train_acc, val_acc = train_model(model_2out, train_dataloader, test_dataloader,
                                                                   val_dataloader, loss,
                                                                   optimizer, scheduler, num_epoch,
                                                                   PATH_TO_SAVE_WEIGHTS)
    # input_image = cv2.imread(
    #     "D:\\IF\\project_bacteria_recognition\\split_2021_2022\\test\\Acinetobacter baumannii\\1.jpeg")

    # visualize_feature_map(model_2out(input_image))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
