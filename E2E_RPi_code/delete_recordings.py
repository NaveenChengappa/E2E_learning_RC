# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:54:17 2020

@brief : Deletes all images and files generated
@author: Naveen Chengappa
"""

import os
import glob

"""
Path Init
"""
image_path = '/home/pi/E2E_learning_RT/cam_images/*.jpg'
image_csv_path = '/home/pi/E2E_learning_RT/CSV_files/image_names.csv'
rover_data_path = '/home/pi/E2E_learning_RT/CSV_files/steering_throttle_data.csv'
output_csv_path = "/home/pi/E2E_learning_RT/CSV_files/output.csv"

"""
Deletes steering-throttle csv file
"""
if os.path.exists(rover_data_path):
  os.remove(rover_data_path)
  print("The file for Rover Data has been deleted.\n")
else:
  print("The file for Rover Data does not exist.")

"""
Deletes images csv file
Deletes all captured images in images folder
"""
if os.path.exists(image_csv_path):
  os.remove(image_csv_path)
  files = glob.glob(image_path)
  for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print('Error in deleting %s : %s' %(f, e.strerror))
  print("The image csv file and all captured images have been deleted.\n")   
else:
  print("The files for images does not exist.")

"""
Deletes output csv file
"""
if os.path.exists(output_csv_path):
  os.remove(output_csv_path)
  print("The output csv file has been deleted.")
else:
  print("The output csv file does not exist.")
