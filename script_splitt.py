import os
import glob
import shutil
from argparse import Namespace

import numpy as np
import random

import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
#
# parser.add_argument('--root', type=str, default='all_files')
# parser.add_argument('--output', type=str, default='all_files_split')
# parser.add_argument('--proportion', type=float, default=0.15)

output = 'D:\\Projects\\bacteria_recognitions\\other_dataset_split'
root = 'D:\\Projects\\bacteria_recognitions\\others_datasets'
proportion = 0.15
if not os.path.exists(output):
    os.mkdir(output)

test_out_folder = os.path.join(output, 'test')
train_out_folder = os.path.join(output, 'train')
if not os.path.exists(test_out_folder):
    os.mkdir(test_out_folder)

if not os.path.exists(train_out_folder):
    os.mkdir(train_out_folder)

subfolders = glob.glob(os.path.join(root, '*'))

for folder in subfolders:

    files = glob.glob(os.path.join(folder, '*'))
    print(f'folder {folder}')
    print(f"number of files {len(files)}")
    files_n = len(files)
    test_n = int(files_n * proportion)
    train_n = files_n - test_n

    test_subfolder = os.path.join(test_out_folder, os.path.basename(folder))
    train_subfolder = os.path.join(train_out_folder, os.path.basename(folder))
    if not os.path.exists(test_subfolder):
        os.mkdir(test_subfolder)
    if not os.path.exists(train_subfolder):
        os.mkdir(train_subfolder)

    test_indexes = random.sample(range(files_n), test_n)

    train_indexes = []
    for i in range(files_n):
        if i not in test_indexes:
            train_indexes.append(i)

    print(
        f'number of train/test = {len(train_indexes)}/{len(test_indexes)}'
    )
    for index in train_indexes:
        file = files[index]
        new_file = os.path.join(train_subfolder, os.path.basename(file))
        shutil.copyfile(file, new_file)

    for index in test_indexes:
        file = files[index]
        new_file = os.path.join(test_subfolder, os.path.basename(file))
        shutil.copyfile(file, new_file)

    pass