import numpy as np
from pylab import *
Path_to_total_mean = "D:\\IF\\total_mean_std_Resnet.txt"
Path_to_save_mean_std_txt = "D:\\IF\\total_mean_std_all_exp.txt"
def total_exp_mean_std(Path, Path_to_save_mean_std_txt):
    mean_ = 0
    i = 0
    list_mean = []
    with open(Path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            mean_ += float(line)
            list_mean.append(float(line))
            i += 1
        mean_acc_test = mean_ / i
        b = np.std(list_mean)
        print(f'mean = {mean_acc_test}, std = {b}')
        with open(Path_to_save_mean_std_txt, 'a+') as file:
            file.write(f"exp {Path}, std test= {b}, mean_acc_test = {mean_acc_test} \n")
# total_exp_mean_std(Path_to_total_mean, Path_to_save_mean_std_txt)

Path = "D:\\IF\\total_mean_std_all_exp.txt"
test_acc = []
mean_acc_test = []
with open(Path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # print('1')
        start = line.index("test=")
        start_ = line.index("mean_acc_test")
        # print(start)
        end = line.index(", mean_acc_test")
        end_ = line.index("\n")
        # print(line[start + 1:end - 1])
        test_acc.append(line[start + 6:end - 1])
        mean_acc_test.append(line[start_ + 16:end_-1])

test_acc_np = list(map(float, test_acc))
mean_acc_test = list(map(float, mean_acc_test))
a_std = test_acc_np
b_mean = mean_acc_test
kol_exp = np.arange(len(b_mean))
# plot(mean_acc_test, b_mean, 'r-')  # 'r' means with red color and '-' means with dashed line style
# bar(a, b, color='red')
# It is ready at this point, but we would like to add error bars:
e = b_mean#0.2 * abs(randn(1))  # prepare errorbar from random numbers

errorbar(kol_exp, mean_acc_test, a_std, fmt='.', label='y=std test(x)')  # vertical symmetric

# Now, we plan to add a legend and axes labels:

legend()  # the label command in errorbar contains the legend

xlabel('x')
xlabel('y')

show()