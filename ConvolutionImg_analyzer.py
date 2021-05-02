# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 01:06:36 2021

@brief : Conv image analyzer
@author: Naveen Chengappa
"""

import matplotlib.pyplot as plt
import numpy as np
import keract
import cv2
from keract import get_activations
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

def load_image(image_name):
    image = Image.open(image_name)
    case1 = image.mode == 'RGBA'
    case2 = image.mode == 'CMYK'
    if case1 or case2:
        image = image.convert('RGB')
    image = img_to_array(image)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image

# Load your model
model = load_model('Dataset/models/e40_spe6k_bs64_split0.2/model-ds1-036-0.008626.h5')
# Load sample image
image = load_image('Dataset/images/02_09__23_31_52_43.jpg')

# Conv image analysis and print
activations = get_activations(model, image, auto_compile=True)
keract.display_activations(activations)

# Heatmap analysis 
#keract.display_heatmaps(activations, image, save=False)

plt.show()

# Steering prediction
steer = model.predict(image)
print("Predicted steering: ", float(steer[0][0]))
