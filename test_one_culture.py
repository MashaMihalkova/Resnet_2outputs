import torch
import torchvision
from torchvision import models, transforms
from tqdm import tqdm

import os.path
import sys
import logging
from logging import error, info
from importlib import reload

import csv
import datetime
import optparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

##
from sklearn import metrics
import itertools


def load_model(_num_classes, _weights_path, _device):
    _model = models.resnet18()
    _model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7),  # для ч/б
                                   stride=(2, 2), padding=(3, 3), bias=False)

    _model.fc = torch.nn.Linear(_model.fc.in_features, _num_classes)
    # _model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(_model.fc.in_features, 256),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.4),
    #     torch.nn.Linear(256, _num_classes),
    #     torch.nn.LogSoftmax(dim=1))

    _model.load_state_dict(torch.load(_weights_path, map_location=torch.device(_device)))
    _model.to(_device)

    return _model


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# def load_images(_transforms, _dataset_path, _batch_size={}, _num_workers=1):
def load_images(_transforms, _dataset_path, _batch_size, _num_workers=1):
    # _test_dataset = torchvision.datasets.ImageFolder(_dataset_path, _transforms)

    _test_dataset = ImageFolderWithPaths(_dataset_path, _transforms)
    # _test_dataloader = {}
    # for class_ in _test_dataset.classes:
    _test_dataloader = torch.utils.data.DataLoader(
        # _test_dataset[class_],
        # batch_size=_batch_size[class_],
        _test_dataset,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=_num_workers)
        # break

    return _test_dataset, _test_dataloader


def tensors2array_classes(_tensors_classes):
    arr_predicted = []  # массив, где каждое число - предсказанный класс
    for batch in range(len(_tensors_classes)):  # по батчам (наборам из нескольких картинок)
        batch_of_samples = _tensors_classes[batch].cpu().numpy()
        for sample in range(len(_tensors_classes[batch])):  # по всем картинкам в тензоре
            arr_predicted.append(batch_of_samples[sample])
    return arr_predicted


def test_model(_model, _test_dataloader, _device):
    arr_tensors_classes = []  # массив тензоров, где в каждом тензоре массив из BATCH_SIZE_TEST чисел, указывающих на класс
    _model.eval()
    for inputs, labels, paths in tqdm(_test_dataloader):
        inputs = torch.unsqueeze(inputs, dim=1)
        inputs = inputs.to(_device)
        # labels = labels.to(_device)
        with torch.set_grad_enabled(False):
            preds = _model(inputs)
            arr_tensors_classes.append(torch.argmax(preds, dim=1))
    mean_prediction = [np.array(arr_tensors_classes[batch]) for batch in range(len(arr_tensors_classes))]
    mean_prediction = np.array(mean_prediction)
    mean_prediction = np.hstack(mean_prediction).tolist()
    mean_prediction_max_freq = max(set(mean_prediction), key=mean_prediction.count)

    predicted_classes = tensors2array_classes(arr_tensors_classes)  # массив предсказанных классов

    return predicted_classes, mean_prediction, mean_prediction_max_freq


def print_results_csv(_filename, full=True):
    if full:
        with open(_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File path',
                             'Class',
                             'Prediction, id',
                             'Prediction, class'])
            for i in range(len(predicted_classes)):
                writer.writerow([test_dataset[i][2],
                                 test_dataset[i][1],
                                 predicted_classes[i],
                                 classes_ids_dict[predicted_classes[i]]])
    else:
        with open(_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File path',
                             'Prediction, class'])
            for i in range(len(predicted_classes)):
                writer.writerow([test_dataset[i][2],
                                 classes_ids_dict[predicted_classes[i]]])


def print_results_as_table(rowsNum=10):
    if rowsNum < 1:
        return
    if rowsNum > len(predicted_classes):
        rowsNum = len(predicted_classes)

    columnsName = ["File path", "Prediction class"]
    w1 = len('../' + '/'.join(test_dataset[0][2].split('/')[-3:])) + 10
    w2 = len(classes_ids_dict[predicted_classes[0]]) + 5

    header = "{}| {}".format(columnsName[0].center(w1), columnsName[1].center(w2))
    line = []
    line.extend(['-' for i in range(len(header))])
    line = "".join(line)
    print(line)
    print(header)

    for row in range(rowsNum):
        trunc_path = '../' + '/'.join(test_dataset[row][2].split('/')[-3:])
        class_str = classes_ids_dict[predicted_classes[row]]
        print("{}| {}".format(trunc_path.ljust(w1), class_str.ljust(w2)))

    print(line)


def show_image(id=0):
    img = mpimg.imread(test_dataset[id][2])  # path to image
    p = plt.imshow(img)
    plt.axis('off')
    trunc_path = '../' + '/'.join(test_dataset[id][2].split('/')[-3:])
    plt.gcf().text(0.02, 1, f'Image Path: {trunc_path}', fontsize=12)
    plt.gcf().text(0.02, 0.95, f'Predicted calss: {classes_ids_dict[predicted_classes[id]]}', fontsize=12)
    plt.show()


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    #       plt.figure(figsize=(10,10))
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    #       plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=0)
    plt.yticks(tick_marks, df_confusion.index)
    #       plt.tight_layout()
    #       plt.tick_params(axis='x', which='major', labelsize=3)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    #       plt.grid()
    plt.savefig('./imgs/cm2.png')


def plot_confusion_matrix1(cm, classes,
                           normalize=True,
                           title='Предсказанный класс',
                           cmap=plt.cm.Blues, path_to_save_img: str = '') -> None:
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': 10})
    #     plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 1) * 100
        #         cm = [np.round(x for x in cm)]
        #         cm *= 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")  # ,
    #              color="white" if cm[i, j] > thresh else "black")

    #     fig = plt.figure(figsize=(20, 20))

    #     plt.tight_layout()
    #     plt.ylabel('Целевой класс')
    # #     plt.xlabel('Предсказанный класс')
    #     plt.savefig('./imgs/cm_test.png')

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(path_to_save_img + '\\cm_test.png', dpi=100)


def plot_confusion_matrix_norm(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues,
                               path_to_save_img: str = '') -> None:
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    #     cm = df_confusion.astype('float') / df_confusion.sum(axis=1)[:, np.newaxis]
    cm = np.round(df_confusion, 1) * 100
    #         cm = [np.round(x for x in cm)]
    #         cm *= 100

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")  # ,
    #              color="white" if cm[i, j] > thresh else "black")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(path_to_save_img + '\\cm_norm.png', dpi=100)


def plot_confusion_matrix_norm1(cm, classes,
                                normalize=False,
                                title='Предсказанный класс',
                                cmap=plt.cm.Blues, path_to_save_img: str = '') -> None:
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': 10})
    #     plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 1) * 100
        #         cm = [np.round(x for x in cm)]
        #         cm *= 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")  # ,
    #              color="white" if cm[i, j] > thresh else "black")

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(path_to_save_img + '\\cm_norm.png', dpi=100)


classes_ids_dict = {0: 'Acinetobacter baumannii',
                    1: 'Acinetobacter pittii',
                    2: 'Acinetobacter seifertii',
                    3: 'Acinetobacter ursingii',
                    4: 'Candida albicans',
                    5: 'Citrobacter braakii',
                    6: 'Citrobacter freundii',
                    7: 'Enterobacter cloacae',
                    8: 'Enterococcus faecalis',
                    9: 'Enterococcus faecium',
                    10: 'Escherichia coli',
                    11: 'Klebsiella aerogenes',
                    12: 'Klebsiella pneumoniae',
                    13: 'Klebsiella variicola',
                    14: 'Proteus mirabilis',
                    15: 'Pseudomonas aeruginosa',
                    16: 'Staphylococcus aureus',
                    17: 'Staphylococcus epidermidis',
                    18: 'Staphylococcus haemolyticus',
                    19: 'Staphylococcus hominis',
                    20: 'Streptococcus agalactiae',
                    21: 'Streptococcus anginosus',
                    22: 'Streptococcus pneumoniae'}

classes_ids_dict_meta = {0: 0,  # 'Acinetobacter',
                         1: 0,
                         2: 0,
                         3: 0,
                         4: 1,  # 'Candida',
                         5: 2,  # 'Citrobacter',
                         6: 2,
                         7: 3,  # 'Enterobacter',
                         8: 4,  # 'Enterococcus',
                         9: 4,
                         10: 5,  # 'Escherichia',
                         11: 6,  # 'Klebsiella',
                         12: 6,
                         13: 6,
                         14: 7,  # 'Proteus',
                         15: 8,  # 'Pseudomonas',
                         16: 9,  # 'Staphylococcus',
                         17: 9,
                         18: 9,
                         19: 9,
                         20: 10,  # 'Streptococcus',
                         21: 10,
                         22: 10}

# ------------------------------------------------------------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS_PATH = 'D:\\Projects\\bacteria_recognitions\\saved_weights\\Exp_Resnet_2outputs_tensor\\ResNet\\Resnet_orig\\REPLAY_seed5_ResNet_input-size_1024\\21'
# TEST_DATASET_PATH = 'D:\\Projects\\bacteria_recognitions\\test_one_shtamm'
TEST_DATASET_PATH = 'D:\\Projects\\bacteria_recognitions\datasets\\total_2021_2022\\test'

INPUT_SHAPE = (1024, 1024)  # (1000, 749)###########################################(1150,862)#(704, 704)
NUM_CLASSES = 23  #################################len(classes_ids_dict)

OUTPUT_RESULTS = 0
VISUALIZE = 0

test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize([0.485], [0.5])
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

logging.shutdown()
reload(logging)
logging.basicConfig(format="%(asctime)s | %(message)s",
                    level=logging.DEBUG,
                    datefmt='%H:%M:%S')

if torch.cuda.is_available():
    info(f'Using GPU: {torch.cuda.get_device_name(0)}.')
else:
    info('GPU unavailable, using CPU instead.')

if not os.path.isfile(MODEL_WEIGHTS_PATH) or not os.stat(MODEL_WEIGHTS_PATH).st_size:
    print(MODEL_WEIGHTS_PATH)
    error("Weights not found. Unable to load the model.")

if not os.path.isdir(TEST_DATASET_PATH):
    error("Dataset directory not found. Unable to load images.")

else:
    number_of_files = sum([len(files) for r, d, files in os.walk(TEST_DATASET_PATH)])
    info(f'Found {number_of_files} files in {TEST_DATASET_PATH}.')

try:
    model = load_model(NUM_CLASSES,
                       MODEL_WEIGHTS_PATH,
                       device)

    info(f'Loaded model: {MODEL_WEIGHTS_PATH}.')
    BATCH_SIZE_TEST = {}
    for _, dirs, files in os.walk(TEST_DATASET_PATH):
        for dir in dirs:
            for _, _, files_dir in os.walk(TEST_DATASET_PATH + '\\' + dir):
                BATCH_SIZE_TEST[dir] = len(files_dir)
        break

    test_dataset, test_dataloader = load_images(test_transforms,
                                                TEST_DATASET_PATH,
                                                _batch_size=BATCH_SIZE_TEST['Acinetobacter baumannii'],
                                                _num_workers=0)


    kol = 0
    for class_ in classes_ids_dict.values():
        # k = test_dataloader[i]
        print(f'dir = {class_} \n Loaded {BATCH_SIZE_TEST[class_]} images.')
        targets = []
        test_dataloader_one_class = []
        if class_ == 'Acinetobacter baumannii':
            for ii in range(BATCH_SIZE_TEST[class_]):
                targets.append(test_dataset.imgs[ii][1])
                test_dataloader_one_class.append(test_dataloader.dataset[ii])
        else:
            for ii in range(kol, kol+BATCH_SIZE_TEST[class_]):
                targets.append(test_dataset.imgs[ii][1])
                test_dataloader_one_class.append(test_dataloader.dataset[ii])
        kol += BATCH_SIZE_TEST[class_]
        predicted_classes, mean_pred_classes, mean_prediction_max_freq = test_model(model, test_dataloader_one_class, device)

        print_results_as_table(OUTPUT_RESULTS)


        # for i1 in range(len(test_dataloader_one_class)):

        # print(np.sum(np.asarray(targets) == np.asarray(predicted_classes)))
        # print(np.sum(np.asarray(targets) == np.asarray(predicted_classes)) / len(predicted_classes))
        print('mean_prediction = ', mean_pred_classes)
        print('mean_prediction_max_freq = ', mean_prediction_max_freq)
        print('target_classes = ', targets[0])
        print('acc mean_pred_classes = ', np.sum(np.asarray(targets[0]) == np.asarray(mean_prediction_max_freq)))

        targets_new = []
        predicted_new = []
        for i1 in range(len(targets)):
            targets_new.append(classes_ids_dict_meta[targets[i1]])
            predicted_new.append(classes_ids_dict_meta[predicted_classes[i1]])

        print("metaklass_acc = ",np.sum(np.asarray(targets_new) == np.asarray(predicted_new)) / len(predicted_new))

        y_actu = pd.Series(targets, name='Target')
        y_pred = pd.Series(predicted_classes, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Target'], colnames=['Predicted'])
        df_conf_norm = df_confusion / df_confusion.sum(axis=0)

        score = metrics.accuracy_score(targets, predicted_classes)
        print("accuracy:   %0.3f" % score)

        # cm = metrics.confusion_matrix(targets, predicted_classes)
        # plot_confusion_matrix1(cm, classes=classes_ids_dict)
        #
        # plot_confusion_matrix_norm(df_conf_norm)

        # break
        # break

except Exception as e:
    error(str(e))









