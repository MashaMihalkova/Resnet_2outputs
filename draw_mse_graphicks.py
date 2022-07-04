import numpy as np
from pylab import *

import pylab as pl
import numpy as np


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Path_to_total_mean = "D:\\Projects\\bacteria_recognitions\\max_Resnet_2outputs_RM_DP_Lin.txt"
# Path_to_save_mean_std_txt = "D:\\Projects\\bacteria_recognitions\\total_max_all_exp.txt"
Path_to_total_mean = "D:\\Projects\\bacteria_recognitions\\saved_weights\\graphiks\\mean_5\\mean_std_Resnet_dropout_mean.txt"
Path_to_save_mean_std_txt = "D:\\Projects\\bacteria_recognitions\\saved_weights\\graphiks\\mean_5\\total_mean_all_exp.txt"

def total_exp_mean_std(Path, Path_to_save_mean_std_txt):
    mean_ = 0
    i = 0
    list_mean = []
    with open(Path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line == '\n':
                continue
            mean_ += float(line)
            list_mean.append(float(line))
            i += 1
        mean_acc_test = mean_ / i
        b = np.std(list_mean)
        print(f'mean = {mean_acc_test}, std = {b}')
        with open(Path_to_save_mean_std_txt, 'a+') as file:
            file.write(f"exp {Path}, std test= {b}, mean_acc_test = {mean_acc_test} \n")
#создать для общего файла
# total_exp_mean_std(Path_to_total_mean, Path_to_save_mean_std_txt)
#
#рисовать errorbar
# Path = "D:\\Projects\\bacteria_recognitions\\saved_weights\\graphiks\\mean_5\\total_mean_all_exp.txt"
# test_acc = []
# mean_acc_test = []
# with open(Path, 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         # print('1')
#         start = line.index("test=")
#         start_ = line.index("mean_acc_test")
#         # print(start)
#         end = line.index(", mean_acc_test")
#         end_ = line.index("\n")
#         # print(line[start + 1:end - 1])
#         test_acc.append(line[start + 6:end - 1])
#         mean_acc_test.append(line[start_ + 16:end_-1])
#
# test_acc_np = list(map(float, test_acc))
# mean_acc_test = list(map(float, mean_acc_test))
# a_std = test_acc_np
# b_mean = mean_acc_test
# kol_exp = np.arange(len(b_mean))
# # plot(mean_acc_test, b_mean, 'r-')  # 'r' means with red color and '-' means with dashed line style
# # bar(a, b, color='red')
# # It is ready at this point, but we would like to add error bars:
# e = b_mean#0.2 * abs(randn(1))  # prepare errorbar from random numbers
#
# errorbar(kol_exp, mean_acc_test, a_std, fmt='.', label='y=std test(x)')  # vertical symmetric
#
# # Now, we plan to add a legend and axes labels:
#
# legend()  # the label command in errorbar contains the legend
#
# xlabel('x')
# xlabel('y')
#
# show()

print('Versions, mpl: {:}, pd: {:}, np: {:}'.format(mpl.__version__, pd.__version__, np.__version__))

df = pd.DataFrame()
df['a'] = [0.3163875365257263 ,
0.3595916497707367 ,
0.34732635140419005 ,
0.3454986834526062 ,
0.34680660486221315 ,
0.3555879199504852 ,
0.3443649756908417
]
df['b'] = [0.3493110728263855 ,
0.3704747724533081 ,
0.36196188926696776 ,
0.34746056914329526 ,
0.3519354689121246 ,
0.34746056199073794 ,
0.3611651599407196
]
df['c'] = [
0.3869629814511254,
0.3646919548511505,
0.36890127658843996,
0.3633868956565857,
0.35471408009529115,
0.35089598417282103,
0.3471835684776306,
]

x_labels = [
    "ResNet-18",
    "ResNet-18-dropout",
    "ResNet_2hears"
]
# x_labels = [
#    0,
#     1,
#     2
# ]
print(df)
# np.arange(df.shape[1]-0.5)
plt.bar(np.arange(3), df.mean(), yerr=[df.mean()-df.min(), df.max()-df.mean()], capsize=10)
# plt.grid()
plt.ylabel('acc')
plt.xlabel('exp')
# plt.set_xticklabels(x_labels)
plt.xticks(x_labels)
plt.show()

#
# x = np.arange(4)
# y1, y2 = [1,2,1,1], [2,3,1,1.5]
#
#
# pl.bar(x+0.2,y2, width=0.45, color='g', zorder=1)
# _, caplines, _ = pl.errorbar(x+0.4,y2,fmt=None, yerr=0.75, ecolor='r', lw=2, capsize=10., mew = 3, zorder=2)
#
# pl.bar(x,y1,width=0.45, zorder=3)
# pl.errorbar(x+0.2,y1,fmt=None, yerr=0.5, ecolor='r',
#             lw=2, capsize=10., mew = 3, zorder=4)
#
# for capline in caplines:
#     capline.set_zorder(2)
#
# pl.savefig('err.png')
# pl.show()


# import numpy as np
# from pylab import *
# Path = "D:\\Projects\\bacteria_recognitions\\saved_weights\\mean_std.txt"
# test_acc = []
# mean_acc_test = []
# with open(Path, 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         # print('1')
#         start = line.index("test=")
#         start_ = line.index("mean_acc_test")
#         # print(start)
#         end = line.index(", mean_acc_test")
#         end_ = line.index("\n")
#         # print(line[start + 1:end - 1])
#         test_acc.append(line[start + 6:end - 1])
#         mean_acc_test.append(line[start_ + 16:end_-1])
#
# test_acc_np = list(map(float, test_acc))
# mean_acc_test = list(map(float, mean_acc_test))
# a_std = test_acc_np
# b_mean = mean_acc_test
# kol_exp = np.arange(len(b_mean))
# # plot(mean_acc_test, b_mean, 'r-')  # 'r' means with red color and '-' means with dashed line style
# # bar(a, b, color='red')
# # It is ready at this point, but we would like to add error bars:
# e = b_mean#0.2 * abs(randn(1))  # prepare errorbar from random numbers
#
# errorbar(kol_exp, mean_acc_test, a_std, fmt='.', label='y=std')  # vertical symmetric
#
# # Now, we plan to add a legend and axes labels:
#
# legend()  # the label command in errorbar contains the legend
#
# xlabel('x')
# xlabel('y')
#
# show()