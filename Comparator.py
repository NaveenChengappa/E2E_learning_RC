# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:43:17 2021

@brief : Training model Validator
@author: Naveen Chengappa
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import converter as conv

def load_image(image_name):
    image = Image.open(image_name)
    case1 = image.mode == 'RGBA'
    case2 = image.mode == 'CMYK'
    if case1 or case2:
        image = image.convert('RGB')
    image = img_to_array(image)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)           # convert to YUV
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    return image

# load CSV file of output data
data_df = pd.read_csv('Dataset\output.csv')
image_files = data_df['file_names'].values
steering = data_df['Norm_angles'].values

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="Dataset\models\e40_spe6k_bs64_split0.2\model-ds1-036-008626.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# model = load_model('model-ds1-040-0.019185.h5')

# set number of samples to be analysed
reqd_data = 2000
# reqd_data = len(data_df)

# set threshold(in degrees) to classify errors value
limit = 5

time = np.arange(0,reqd_data)
actual_steer = []
predict_steer = []
error = []
img_path = "Dataset/images/"
for i in range(reqd_data):
    image = load_image(img_path+image_files[i])
    input_data = np.array(image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    steer = interpreter.get_tensor(output_details[0]['index'])
    steer = float(steer[0][0])
    actual_steer.append(conv.pwm_to_angle(conv.norm_angle_to_pwm(steering[i])))
    predict_steer.append(conv.pwm_to_angle(conv.norm_angle_to_pwm(steer)))
    error.append(round(actual_steer[i] - predict_steer[i]))      
    # print("Actual steering   : ", (conv.pwm_to_angle(conv.norm_angle_to_pwm(steering[i]))))
    # print("Predicted steering: ", conv.pwm_to_angle(conv.norm_angle_to_pwm(steer)))
    # print("Absolute difference: ", abs(steering[i] - steer))
    # print("\n")
  
# Actual vs. predicted value plot
plt.plot(time, actual_steer, 'r-', label='Actual')    
plt.plot(time, predict_steer, 'b-', label='Predicted')   
plt.grid(True)  
plt.legend()
plt.ylabel('steering angle in degrees')
plt.xlabel('image sample number')
plt.show()

# Error disctribution plot
fig, ax = plt.subplots()
ax.set_title('Error plot', fontsize=15)
ax.set_xlabel("steering error (in degrees)", fontsize=15)
ax.set_ylabel("value distribution", fontsize=15)
tempy, tempx, _ = ax.hist(error, bins=20, color='dimgrey')
ax.grid(True) 

# Analysis
print('***** Prediction data analysis *****')
print("Total samples analysed :", reqd_data)
print("Samples with 0 degree error : ", len([x for x in error if x==0]))
print("Samples with upto 5 degrees error : ", len([y for y in error if y<limit and y>-limit and y !=0]))
print("Samples with greater than 5 degrees error : ", len([z for z in error if z>limit or z<-limit]))



