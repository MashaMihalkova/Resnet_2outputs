# from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# from processor import process_image
# from keras.models import load_model
from pylab import *
import torch
import cv2


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print("888888=", feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        title('feature_map_{}'.format(i))

    plt.savefig('feature_map.jpg')
    plt.show()

    # Каждая карта функций накладывается 1: 1
    feature_map_sum = sum(ele for ele in feature_map_combination)
    #    feature_map_sum=(feature_map_sum-np.min(feature_map_sum))/(np.max(feature_map_sum)-np.min(feature_map_sum))
    y_predict = np.array(feature_map_sum).astype('float')
    y_predict = np.round(y_predict, 0).astype('uint8')
    y_predict *= 255
    y_predict = np.squeeze(y_predict).astype('uint8')
    cv2.imwrite("C:\\Users\\Administrator\\Desktop\\0.tif", y_predict)

    plt.imshow(y_predict)
    plt.savefig("y_predict.jpg")


def visualize_feature_map1(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print("feature_map_shape=", feature_map.shape)
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    for _ in range(num_pic):
        show_img = f1[:, :, :, _]
        show_img.shape = [feature_map.shape[0], feature_map.shape[1]]
        plt.subplot(row, col, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')

    plt.show()
    plt.savefig("4444.jpg")

#
# if __name__ == "__main__":
#     # replaced by your model name
#     model = load_model("C:\\Users\\Administrator\\Desktop\\9696.hdf5")
#     ## Первый model.layers [0], не изменять, представляет входные данные; второй model.layers [вы хотели], изменяет количество слоев, которые необходимо вывести.
#     layer_1 = K.function([model.layers[0].input], [model.layers[67].output])
#     input_image = np.load("D:\\IF\\project_bacteria_recognition\\split_2021_2022\\test\\Acinetobacter baumannii\\1.jpeg")
#     f1 = layer_1([input_image])[0]  # Только изменять inpu_image
#     # Первый слой отображения карты объектов после свертки, вывод будет (1,149,149,32), (количество образцов, размер карты объектов длинный, размер карты объектов широкий, номер карты объектов)
#     visualize_feature_map(f1)
