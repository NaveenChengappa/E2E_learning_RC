# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:54:17 2020

@brief : Code for data collection phase
@author: Naveen Chengappa
"""

import time
import picamera
import datetime
from datetime import datetime
from os import walk
import csv
import pandas as pd
import serial
import os
import threading
import converter as conv
import RPi.GPIO as GPIO
from time import sleep

"""
Get image file names to merge
in CSV with ADC data
"""
def filenames():
    count = 0
    while count < 50:
        yield ('cam_images/' + time.strftime("%y_%m_%d__%H_%M_%S") + '.jpg')
        count += 1

"""
Read from the Arduino serial monitor
and store values with timestamp in CSV file
"""
def adc_reads():
	Time_Stamp = 0
	decoded_throttle = 0.00
	decoded_steering = 0.00
	steering_angle = 0
	try:
		while ledState == True:
			ser_bytes = ser.readline()
			try:
				decoded_steering = float(ser_bytes[0:4].decode("utf-8"))
				decoded_throttle = float(ser_bytes[7:11].decode("utf-8"))
				#steering_angle = conv.pwm_to_angle(decoded_steering)
			except ValueError:
				pass
			Time_Stamp = datetime.utcnow().strftime("%m_%d__%H_%M_%S_%f")[:-4]
			print('{} {} {}' .format(Time_Stamp, decoded_steering, decoded_throttle))
			with open("/home/pi/E2E_learning_RT/CSV_files/steering_throttle_data.csv","a") as f:
				writer = csv.writer(f,delimiter=",")
				writer.writerow([Time_Stamp, decoded_steering,decoded_throttle])
				
	except KeyboardInterrupt:
		pass

"""
Merge camera file names with 
ADC readings in CSV file
"""
def cam_csv_generator():
	f = []
	path = '/home/pi/E2E_learning_RT/cam_images'
	for (dirpath, dirnames, filenames) in walk(path):
		f.extend(filenames)
		break
	df = pd.DataFrame(data={"file_names":f})
	df.to_csv("/home/pi/E2E_learning_RT/CSV_files/image_names.csv",sep=',',index=False)

"""
Capture images with preview
Save images with time stamp as file name
"""
def camera_capture():
	with picamera.PiCamera() as camera:
		camera.resolution = (200, 66)
		camera.framerate = 20
		camera.rotation = 180
		camera.start_preview()
		# Give the camera some warm-up time
		time.sleep(2)    
		try:
			while ledState == True:	
				camera.capture('cam_images/' + (datetime.utcnow().strftime("%m_%d__%H_%M_%S_%f")[:-4]) + '.jpg', use_video_port=True)
		except KeyboardInterrupt:
			time.sleep(2)
			cam_csv_generator()
			pass

############# Main ############

ser = serial.Serial('/dev/ttyACM0')
ser.flushInput()
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

blinkCount = 2   #Set number of training attempts in one session
count = 0
LEDPin = 16      # Green LED
LEDPin_2 = 18    # Red LED
buttonPin = 10
GPIO.setup(LEDPin, GPIO.OUT)
GPIO.setup(LEDPin_2, GPIO.OUT)
GPIO.setup(buttonPin, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
buttonPress = True
ledState = False

try:
    print("Press button to start recording session....")
    while count < blinkCount:
        buttonPress = GPIO.input(buttonPin)
        if buttonPress == True and ledState == False:
            GPIO.output(LEDPin,GPIO.HIGH)
            GPIO.output(LEDPin_2,GPIO.LOW)
            print('Recording In Progress')
            ledState = True
            threading.Thread(target = adc_reads).start()
            threading.Thread(target = camera_capture).start()
            sleep(0.5)
        elif buttonPress == True and ledState == True:
            GPIO.output(LEDPin,GPIO.LOW)
            GPIO.output(LEDPin_2,GPIO.HIGH)
            ledState = False
            count +=1            
            print('Recoding Stopped. Remaining attempts in current session: {}'.format(blinkCount - count))
            sleep(0.5)
        sleep(0.1)

    if count >= blinkCount:
        GPIO.output(LEDPin_2,GPIO.LOW)
        print("Max attempts for session completed !")
        GPIO.cleanup()
        pass

except KeyboardInterrupt:
	time.sleep(1)
	GPIO.cleanup()
	time.sleep(1)
	print("Session terminated by user !")
	pass
