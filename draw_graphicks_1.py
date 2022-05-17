import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import *

def plot_train_history(train, val, type: str, test):
    epochs = range(len(train))
    plt.figure()
    train_acc_np = list(map(float, train))
    mean_acc_train = sum(train_acc_np) / len(train_acc_np)
    val_acc_np = list(map(float, val))
    mean_acc_val = sum(val_acc_np) / len(val_acc_np)
    if test is not None:
        test_acc_np = list(map(float, test))
        mean_acc_test = sum(test_acc_np) / len(test_acc_np)
        print(f"mean_acc_train = {mean_acc_train}, mean_acc_val = {mean_acc_val}, mean_acc_test = {mean_acc_test}")
    else:
        print(f"mean_acc_train = {mean_acc_train}, mean_acc_val = {mean_acc_val}")

    title = type
    plt.plot(epochs, train_acc_np, 'b', label=f'Training {type}')
    plt.plot(epochs, val_acc_np, 'r', label=f'Validation {type}')
    if test is not None:
        plt.plot(epochs, test_acc_np, 'g', label=f'Test {type}')

        # a = test_acc_np
        # b = np.std(a)
        # print("std test= ", b)
        # with open(Path, 'a+') as file:
        #     file.write(f"exp {}, std test= {b}, mean_acc_test = {mean_acc_test}")
        # plot(mean_acc_test, b, 'r-')  # 'r' means with red color and '-' means with dashed line style
        # bar(a, b, color='red')
        # It is ready at this point, but we would like to add error bars:
        # e = b#0.2 * abs(randn(1))  # prepare errorbar from random numbers
        # errorbar(0, mean_acc_test, e, fmt='.', label='y=std test(x)')  # vertical symmetric

        # Now, we plan to add a legend and axes labels:

        # legend()  # the label command in errorbar contains the legend
        #
        # xlabel('x')
        # xlabel('y')

        # show()
    plt.title(title)
    plt.legend()

    plt.show()


# Path = "D:\\IF\\graphiks\\exp_Lera_3.txt"
# Path_test = ''  # "D:\\IF\\graphiks\\exp_Lera_2_test.txt"
# Path = "D:\\IF\\graphiks\\ResNet_2out_MaxPooling_after4_sum0.3_input-size_1024.txt"
Path = "D:\\IF\\graphiks\\RM_linear_dropout\\RM_linear_and_dropout_seed300_ResNet_2output_after3layer_weight_decay1_summ0.3+1_input-size_1024.txt"
# Path = "D:\\Projects\\bacteria_recognitions\\saved_weights\\Exp_Resnet_total\\Resnet_RM_dropout\\seed99_Resnet_input_b23_wd5_q0_u1-size_1024\\otchet.txt"
Path_to_mean_std_txt = "D:\\IF\\mean_std_Resnet_2outputs_RM_DP_LIN.txt"
# Path = "D:\\Projects\\bacteria_recognitions\\NEW_DATA\\RESNET\\EXP_BS\\seed895_Resnet_input_b16_wd5_q0_u0-size_1024.txt"
if "exp_Lera" in Path:
    Path_test = Path[:-4]+"_test.txt"
    flag_resnet_2output = 0
elif "Original_ResNet" in Path:# or "ResNet_input":
    Path_test = ''
    flag_resnet_2output = 0
else:
    Path_test = ''
    flag_resnet_2output = 1


train_loss = []
train_acc = []
val_loss = []
val_acc = []
test_loss = []
test_acc = []
out2_train = []
out2_val = []
out2_test = []
train_loss_2out = []
train_acc_2out = []
val_loss_2out = []
val_acc_2out = []
test_loss_2out = []
test_acc_2out = []
with open(Path, 'r') as file:
    i = -1
    lines = file.readlines()
    for line in lines:
        i += 1
        # for n, line in enumerate(file, 1):
        #     line = line.rstrip('\n')
        if 'exp_Lera' in file.name:
            if line.find("train Loss") != -1:
                # print('1')
                start = line.index(":")
                # print(start)
                end = line.index("Acc:")
                end_ = line.index("\n")
                # print(line[start + 1:end - 1])
                train_loss.append(line[start + 1:end - 1])
                if end_:
                    train_acc.append(line[end + 5:end_])
                else:
                    train_acc.append(line[end + 5:])
            if line.find("val Loss") != -1:
                # print('1')
                start = line.index(":")
                # print(start)
                end = line.index("Acc:")
                end_ = line.index("\n")
                # print(line[start + 1:end - 1])
                val_loss.append(line[start + 1:end - 1])
                if end_:
                    val_acc.append(line[end + 5:end_])
                else:
                    val_acc.append(line[end + 5:])

        else:
            if flag_resnet_2output:
                if line.find("train") != -1:
                    # file.__next__()
                    # line.next()
                    l = lines[i + 1]
                    start = l.index("Loss:")
                    # print(start)
                    end = l.index("Acc:")
                    start_2 = l.index("1 OUTPUT:")
                    # print(l[start + 5:end - 1])
                    train_loss.append(l[start + 5:end - 1])
                    train_acc.append(l[end + 5:start_2])
                    out2_train.append(l[start_2 + 14:])
                    # train_acc_2out.append(l)

                    # print(l)
                if line.find("val") != -1:
                    # file.__next__()
                    # line.next()
                    l = lines[i + 1]
                    start = l.index("Loss:")
                    # print(start)
                    end = l.index("Acc:")
                    start_2 = l.index("1 OUTPUT:")
                    # print(l[start + 5:end - 1])
                    val_loss.append(l[start + 5:end - 1])
                    val_acc.append(l[end + 5:start_2])
                    out2_val.append(l[start_2 + 14:])
                    # print(l)
                if line.find("test") != -1:
                    # file.__next__()
                    # line.next()
                    l = lines[i + 1]
                    start = l.index("Loss:")
                    # print(start)
                    end = l.index("Acc:")
                    start_2 = l.index("1 OUTPUT:")
                    # print(l[start + 5:end - 1])
                    test_loss.append(l[start + 5:end - 1])
                    test_acc.append(l[end + 5:start_2])
                    out2_test.append(l[start_2 + 14:])

            else:
                if line.find("train") != -1:
                    # file.__next__()
                    # line.next()
                    l = lines[i + 1]
                    start = l.index("Loss:")
                    # print(start)
                    end = l.index("Acc:")
                    # print(l[start + 5:end - 1])
                    train_loss.append(l[start + 5:end - 1])

                    train_acc.append(l[end + 5:])
                    # print(l)
                if line.find("val") != -1:
                    # file.__next__()
                    # line.next()
                    l = lines[i + 1]
                    start = l.index("Loss:")
                    # print(start)
                    end = l.index("Acc:")
                    # print(l[start + 5:end - 1])
                    val_loss.append(l[start + 5:end - 1])

                    val_acc.append(l[end + 5:])
                    # print(l)
                if line.find("test") != -1:
                    # file.__next__()
                    # line.next()
                    l = lines[i + 1]
                    start = l.index("Loss:")
                    # print(start)
                    end = l.index("Acc:")
                    # print(l[start + 5:end - 1])
                    test_loss.append(l[start + 5:end - 1])

                    test_acc.append(l[end + 5:])



if "exp_Lera" in Path_test:
    with open(Path_test, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                end = line.index("\n")
            except Exception:
                print("1")

            if end:
                test_acc.append(line[:end])
            else:
                test_acc.append(line)

if flag_resnet_2output:
    for i in range(len(out2_train)):
        train_ind = out2_train[i].index("Acc:")
        end = out2_train[i].index("\n")
        train_loss_2out.append(out2_train[i][2:train_ind])
        train_acc_2out.append((out2_train[i][train_ind + 5:end]))
        val_ind = out2_val[i].index("Acc:")
        end = out2_val[i].index("\n")
        val_loss_2out.append(out2_val[i][2:val_ind])
        val_acc_2out.append((out2_val[i][val_ind + 5:end]))
        test_ind = out2_test[i].index("Acc:")
        try:
            end = out2_test[i].index("\n")
        except:
            print("1")
        test_loss_2out.append(out2_test[i][2:test_ind])
        test_acc_2out.append((out2_test[i][test_ind + 5:end]))

if not test_loss:
    plot_train_history(train_acc, val_acc, "accuracy", test=test_acc)
    plot_train_history(train_loss, val_loss, "loss", test=None)
else:
    # plot_train_history(train_acc, val_acc, "accuracy", test=test_acc)
    # plot_train_history(train_loss, val_loss, "loss", test=test_loss)

    test_acc_np = list(map(float, test_acc))
    mean_acc_test = sum(test_acc_np) / len(test_acc_np)
    b = np.std(test_acc_np)
    with open(Path_to_mean_std_txt, 'a+') as file:
        file.write(f"exp {Path}, std test= {b}, mean_acc_test = {mean_acc_test} \n")
    # if "Original_ResNet" in Path :#or "ResNet_input":
    #     plot_train_history(train_acc_2out, val_acc_2out, "accuracy", test=test_acc_2out)
    #     plot_train_history(train_loss_2out, val_loss_2out, "loss", test=test_loss_2out)
