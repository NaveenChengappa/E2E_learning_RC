# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:54:17 2020

@brief : Merges collected data and generates output CSV file
@author: Naveen Chengappa
"""

import csv
import pandas as pd
from os import walk
from datetime import datetime
import glob
import os
import converter as conv

my_list1 = []
my_list2 = []
f = []
angle_list = []
norm_angle_list = []

"""
Path Init
"""
image_path = '/home/pi/E2E_learning_RT/cam_images'
image_csv_path = '/home/pi/E2E_learning_RT/CSV_files/image_names.csv'
rover_data_path = '/home/pi/E2E_learning_RT/CSV_files/steering_throttle_data.csv'
output_csv_path = "/home/pi/E2E_learning_RT/CSV_files/output.csv"
backup_output_csv_path = "/home/pi/E2E_learning_RT/backup_output_csv/"

for (dirpath, dirnames, filenames) in walk(image_path):
	f.extend(filenames)
	break
df = pd.DataFrame(data={"file_names":f})
df.sort_values(by = "file_names", inplace = True)
df.to_csv("/home/pi/E2E_learning_RT/CSV_files/image_names.csv",sep=',',index=False)

for root, _, files in os.walk(image_path):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.getsize(fullpath) < 1 * 1024:
            os.remove(fullpath)

df_img = pd.read_csv (image_csv_path)
df_img.set_index('file_names', inplace = True)

for obj in df_img.index:
     my_list1.append(obj[:-5])
df_img['Time stamp'] = my_list1
df_img.reset_index(inplace =True)

mylist = ['Time stamps','steering','throttle']
df_read = pd.read_csv (rover_data_path)
df_read.columns = mylist
df_read.set_index('Time stamps', inplace = True)
for obj in df_read.index:
     my_list2.append(obj[:-1])
df_read['Time stamp'] = my_list2

# Enable angle column in csv
for obj in df_read['steering']:      
    angle_list.append(conv.pwm_to_angle(obj))    
df_read['Angles'] = angle_list

# Enable normalised angle column in csv
for obj in df_read['steering']:      
    norm_angle_list.append(conv.pwm_to_norm_angle(obj))
df_read['Norm_angles'] = norm_angle_list

# Merge 2 csv files
merged = df_img.merge(df_read, on ='Time stamp')
del merged['Time stamp']
merged.to_csv(output_csv_path, index=False)

# Create a back up csv in a different folder
filename1 = datetime.now().strftime("%Y_%m_%d--%H_%M")
merged.to_csv(backup_output_csv_path+filename1+'.csv', index=False)

print('All files generated successfully')

