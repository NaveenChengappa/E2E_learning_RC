# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:24:25 2021

@brief : Training script
@author: Naveen Chengappa

"""
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import datetime
from time import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import img_to_array
import os
import cv2

# Path initializations
main_path = os.path.join(os.path.abspath(""),"Dataset")
img_dir = os.path.join(main_path,"images")
csv_file =  os.path.join(main_path, "output.csv")
model_name = "model-ds1-036-0.008626.h5"
logdir = os.path.join(main_path, "logs")
saved_logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
filepath = os.path.join(main_path, "models", "model-ds1-{epoch:03d}-{val_loss:.6f}.h5")

"""
Training parameters
"""
EPOCHS = 40
spe = 6000
bs = 64
LR = 1.0e-04
test_train_split = 0.2
default_labels = ['image', 'steering', 'throttle', 'angle', 'norm']

# === DEFINE VARIABLES
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def isfile(file):
  return os.path.isfile(file)

def isdir(path):
    return os.path.isdir(path)

# -- Load csv file
def load_rc_csv(csv_file):
    data_df = pd.read_csv(csv_file)
    data_df.columns = default_labels
    data = data_df[default_labels].values
    return data

# -----------------------------------------------------------------------------
# TRAINING SCRIPT -------------------------------------------------------------
# -----------------------------------------------------------------------------

# - @private: Change RGB to YUV
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# - @private: Flip image randomly
def flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    else:
        image = image
        steering_angle = steering_angle

    return image, steering_angle

# - @private: Translate image randomly
def translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, steering_angle


# - @private: Load image (not in use)
def load_image(image_path):
    return mpimg.imread(image_path)

# - @private: Load RGB image
def load_rc_image(image):
    image = os.path.join(img_dir, image)
    image = Image.open(image)
    case1 = image.mode == 'RGBA'
    case2 = image.mode == 'CMYK'
    if case1 or case2:
        image = image.convert('RGB')
    image = img_to_array(image)
    image = image.astype(np.uint8)
    return image

# - @public: Convert image to yuv format
def preprocess1(image):
    image = rgb2yuv(image)
    return image

# - @public: Process image by flipping and translating
def preprocess2(image, steering_angle, range_x=100, range_y=10):
    image = load_rc_image(image)
    image, steering_angle = flip(image, steering_angle)
    image, steering_angle = translate(image, steering_angle, range_x, range_y)

    return image, steering_angle

# - @public: Data generator
def data_generator(feature, label, batch_size, is_training, lucky_number=0.5):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    angles = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(feature.shape[0]):
            image = feature[index]
            steering = label[index]

            if is_training and np.random.rand() < lucky_number:
                image, steering = preprocess2(image, steering)
            else:
                image = load_rc_image(image)

            images[i] = preprocess1(image)
            angles[i] = steering
            i += 1
            if i == batch_size:
                break
        yield images, angles

# - Loading CSV file
def load_data(amount=3):    
    data_df = pd.read_csv(csv_file)
    data_df.columns = default_labels
    x = data_df[default_labels[0]].values # Image file name as input
    y = data_df[default_labels[4]].values # steering value as output
    y = np.ndarray.tolist(y)
    print('Data length: {}'.format(len(x)))    
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_train_split, random_state=0)
    print('Test - Train split :', test_train_split)
    return x_train, x_valid, y_train, y_valid

# - Build model
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

# - Main function
def main(model_name=None):
    # load training data
    x_train, x_valid, y_train, y_valid = load_data()

    # load model
    if model_name is None:
        model = nvidia_model()
    else:
        print("Train from model:", model_name)
        model = load_model(model_name)

    # Callbacks
    # -- ModelCheckpoint
    callback_modelcheckpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')

    # -- Tensorboard
    callback_tensorboard = tf.keras.callbacks.TensorBoard(saved_logdir, histogram_freq=1)                                                      

    # -- Final callbacks
    callbacks = [callback_modelcheckpoint, callback_tensorboard]
    
    # Compile
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LR))
    
    # model fit
    model.fit(data_generator(x_train, y_train, batch_size=bs, is_training=True),
                        steps_per_epoch=spe,
                        epochs=EPOCHS,
                        max_queue_size=1,
                        validation_data=data_generator(x_valid, y_valid, batch_size=bs,
                                                       is_training=False),
                        validation_steps=len(x_valid)/bs,
                        callbacks=callbacks,
                        verbose=1)    

# Histogram for data analysis    
def hisplot_rc_data(y_train, title=None, fs=15, nob=100, y_distance=100):    
    y = [float(item) for item in y_train]
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel("steering values", fontsize=fs)
    ax.set_ylabel("value distribution", fontsize=fs)
    tempy, tempx, _ = ax.hist(y, bins=nob, color='dimgrey')
    ax.set_ylim([0, max(tempy) + y_distance])

# Test code for sample image before training
# Post processing data analysis histogram
def test():
    choice = 0
    x_train, x_valid, y_train, y_valid = load_data(amount=3)
    choice = np.random.randint(0, len(x_train))
    image = x_train[choice]
    steer = y_train[choice]
    print("IMG name :", image)
    print("Steer    :", steer)
    img = load_rc_image(image)
    img = preprocess1(img)
    plt.imshow(img)
    plt.show()
    hisplot_rc_data(y_train, title='Post Processing Data Analysis')

if __name__ == '__main__':
    # Check TF version
    print("Tensorflow version: ", tf.__version__)
    print("\nBEFORE TRAINING ===================================================")
    test()
    print("\nAFTER TRAINING ====================================================")
    main(model_name=None)
