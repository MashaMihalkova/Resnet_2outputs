import numpy as np
import torch.nn as nn
import torch
from sklearn.svm import SVC


class NN_with_SVM(nn.Module):
    '''
    About implementation,
    you just have to train a neural network,
    then select one of the layers
    (usually the ones right before the fully connected layers or the first fully connected one),
     run the neural network on your dataset,
      store all the feature vectors,
    then train an SVM with a different library (e.g sklearn).
    '''
    def __init__(self, layer_number:int = 3, num_classes: int = 1000, color_or_grey: str = "grey") -> None:
        super(NN_with_SVM, self).__init__()


#
# svc = SVC()
# model = svc.fit(X_train,y_train)
# y_pred_svc = model.predict(X_test)
# print("\n Done")
# print("\n Accuracy of SVM: ",accuracy(y_test,y_pred_svc))

