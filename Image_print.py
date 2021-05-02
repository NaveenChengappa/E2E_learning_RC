# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 01:06:36 2021

@brief : Image processing code
@author: Naveen Chengappa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

"""
Path initializations
"""
main_path = os.path.join(os.path.abspath(""),"Dataset")
img_dir = os.path.join(main_path,"images")
csv_file =  os.path.join(main_path, "output.csv")

test_train_split = 0.2
default_labels = ['image', 'steering', 'throttle', 'angle', 'norm']

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

# - Loading CSV file
def load_data():    
    data_df = pd.read_csv(csv_file)
    data_df.columns = default_labels       
    x = data_df[default_labels[0]].values # Image file name as input
    y = data_df[default_labels[4]].values # steering value as output
    y = np.ndarray.tolist(y)   
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, 
                                                          test_size=test_train_split, 
                                                          random_state=0)
    return x_train, x_valid, y_train, y_valid

# - @private: Flip image randomly
def flip(image, steering_angle):      
    if np.random.rand() < 0.5:
        plt.imshow(image)
        plt.title('Image before Flip :: Steering : {}'.format(steering_angle))
        plt.show()
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle 
        plt.imshow(image)
        plt.title('Image after Flip :: Steering : {}'.format(steering_angle))
        plt.show()
    else:
        image = image
        steering_angle = steering_angle
        print("Steering not flipped")
    return image, steering_angle

# - @private: Translate image randomly
def translate(image, steering_angle, range_x, range_y):    
    plt.imshow(image)
    plt.title('Image before translation :: Steering : {}'.format(steering_angle))
    plt.show()
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002 #translation factor
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))    
    plt.imshow(image)
    plt.title('Image after translation :: Steering : {}'.format(round(steering_angle,3)))
    plt.show()
    return image, steering_angle

# - @public: Process image by flipping and translating
def preprocess2(image, steering_angle, range_x=100, range_y=10):
    image = load_rc_image(image)
    image, steering_angle = flip(image, steering_angle)
    image, steering_angle = translate(image, steering_angle, range_x, range_y)
    return image, steering_angle

# Main code
def main():
    choice = 0
    x_train, x_valid, y_train, y_valid = load_data()
    choice = np.random.randint(0, len(x_train))
    image = x_train[choice]
    steering = y_train[choice]
    image, steering = preprocess2(image, steering)

if __name__ == '__main__':
    main()