# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:24:25 2021

@brief : Data analysis code
@author: Naveen Chengappa
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from datetime import datetime
import os
import datetime
import csv

"""
Path initializations
"""
main_path = os.path.join(os.path.abspath(""),"Dataset")
img_dir = "images"
csv_file =  os.path.join(main_path, "output.csv")
model_name = "model-ds1-036-0.008626.h5"
default_labels = ['image', 'steering', 'throttle', 'angle', 'norm']

"""
Training parameters
"""
EPOCHS = 40
spe = 6000
bs = 64
LR = 1.0e-04

def isfile(file):
  return os.path.isfile(file)

# -- Load csv file
def load_rc_csv(csv_file):
    data_df = pd.read_csv(csv_file)
    data = data_df[:].values
    return data

# -- Histogram plot of data
def hisplot_rc_data(csv_file, title=None, fs=15, nob=100, y_distance=100):
    data = load_rc_csv(csv_file)
    data = np.asarray(data)
    y = data[:, -1]
    y = np.ndarray.tolist(y)
    y = [float(item) for item in y]

    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel("steering values", fontsize=fs)
    ax.set_ylabel("value distribution", fontsize=fs)
    tempy, tempx, _ = ax.hist(y, bins=nob, color='dimgrey')
    ax.set_ylim([0, max(tempy) + y_distance])

# -- Count labels
def count_line_csv(csvfile):
    with open(csvfile, 'r', newline='') as file:
        file_object = csv.reader(file)
        row_count = sum(1 for row in file_object)
        return row_count

# -- Count images
def images_count(dir):
    count = len([img for img in os.listdir(dir)])
    return count

# -- Check for duplicate items
def check_duplicate_data(csv_file, display_dup=False):
    print("\nCalling: duplication checking")
    data = load_rc_csv(csv_file)
    img_names = data[:, 0]
    img_names = np.ndarray.tolist(img_names)
    dup_list = []
    dupsum = 0
    for i in range(2, 6):
        temp = []
        for item in img_names:
            if img_names.count(item) == i:
                dup_list.append(item)
                temp.append(item)
        dupsum += (len(temp)/i) * (i - 1)
        print("- Num of dup {}: {}".format(i, len(temp) / i))
    print("- Amount of duplicates: ", dupsum)
    # Display duplication summary
    if display_dup:
        if len(dup_list) > 0:
            for item in dup_list:
                print(item)

# -- Check for balance between features and labels
def data_balance_check(dir, csvfile):
    print("\nCalling: data balance checking")
    img_num = images_count(dir)
    line_num = count_line_csv(csvfile)
    print("- Total number of Images : {}".format(img_num))
    print("- Total number of Samples: {}".format(line_num))
    if img_num == line_num and img_num != 0 and line_num != 0:
        print('- Result: Balanced data.')
        return True
    else:
        print('- Result: Image Duplication present.')
        # Check duplicate items
        check_duplicate_data(csv_file)
        return False

"""
-----------------------------------------------------------------------------
Print training parameters ---------------------------------------------------
-----------------------------------------------------------------------------
"""
print("Tensorflow version: ", tf.__version__)
print("\n-------- TRAINING PARAMETERS --------")
print("         Model: ", model_name)
print("        EPOCHS: ", EPOCHS)
print("    Batch size: ", bs)
print(" Learning Rate: ", LR)
print("Step per epoch: ", spe)

"""
-----------------------------------------------------------------------------
IMAGE SAMPLE CHECK ----------------------------------------------------------
-----------------------------------------------------------------------------
"""
img_dir = os.path.join(main_path, "images")
images = os.listdir(img_dir)
rint = np.random.randint(0, len(images) - 1)
image = mpimg.imread(os.path.join(img_dir, images[rint]))
print("\n-------- Image sample -------------------")
print("Total images :", len(images))
print("Image dtype  :", image.dtype)
print("Image type   :", type(image))
print("Image shape  :", image.shape)
print("Total samples in dataset :", count_line_csv(csv_file))
plt.show()

"""
-----------------------------------------------------------------------------
CSV SAMPLE CHECK ------------------------------------------------------------
-----------------------------------------------------------------------------
"""
data = load_rc_csv(csv_file)
choice = np.random.randint(0, len(data)-1)
test_sample = data[choice]
print("\n-------- CSV data sample ----------------")
print(default_labels[0], "   :", test_sample[0])
print(default_labels[1], ":", test_sample[1])
print(default_labels[2], ":", test_sample[2])
print(default_labels[3], "   :", test_sample[3])
print(default_labels[4], "    :", test_sample[4])


print("\n-------- Data Analysis -----------------")
# Histogram plot
hisplot_rc_data(csv_file, nob=150, title="Pre Processing Data Analysis")
# Data balance check
data_balance_check(img_dir, csv_file)

