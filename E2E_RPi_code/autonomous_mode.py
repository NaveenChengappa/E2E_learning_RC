# -*- coding: utf-8 -*-
"""
Created on Sat March 06 11:54:17 2021

@author: Naveen Chengappa
"""

import time
import picamera
import RPi.GPIO as GPIO
import picamera.array
from tensorflow.keras.models import load_model
import converter as conv
import threading
import pigpio
import tensorflow as tf
import numpy as np

# define input image shape
IMG_H = 66
IMG_W = 200
IMG_C = 3
INPUT_SHAPE = (IMG_H, IMG_W, IMG_C)

# set the speed modes
THR_IDLE = 899
THR_SLOW = 938
THR_FAST = 940

"""
Initialize steering pins
"""
def init_auto_steer(LEDPin,LEDPin_2,buttonPin,servoPIN):
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LEDPin, GPIO.OUT)
    GPIO.setup(LEDPin_2, GPIO.OUT)    
    GPIO.setup (servoPIN, GPIO.OUT)    
    p = GPIO.PWM (servoPIN, 50) # GPIO 17 as PWM with 50Hz
    p.start(0) # initialization
    return p

"""
Initialize throttle pins
"""
def throttle_init(pwm_pin):       
    thr_p = pigpio.pi()    
    thr_p.set_mode(pwm_pin,pigpio.OUTPUT)
    return thr_p

"""
Initialize camera
"""
def cam_init():
    camera = picamera.PiCamera()
    camera.resolution = (IMG_W, IMG_H)
    camera.framerate = 30
    camera.rotation = 180
    camera.start_preview(fullscreen=False,window=(100,200,600,300))
    time.sleep(1)
    return camera

"""
Calibrate Throttle PWM pin
"""
def throttle_calibration(thr_p,pwm_pin,freq):
    print("*****************")
    print("Throttle Calibration")
    print("*****************")
    input("Switch OFF the DC motor and press return key to continue...")
    thr_p.hardware_PWM(pwm_pin,freq, THR_IDLE *100) 
    print("Calibrating - step 1/3.... ")
    time.sleep(3)
    input("Switch ON the DC motor and press return key to continue...")
    print("Calibrating - step 2/3.... ")
    time.sleep(3)
    input("Switch Steering to Autonomous mode and press return key to confirm.")
    print("Calibrating - step 3/3.... ")
    print("Calibration Completed\n")
    return 1

"""
control steering by changing PWM
"""
def auto_steer(angle, p):
    try:
        # my_angle = float((angle + 30)/20) + 5
        pctg = float((2*(angle + 1)) + 5)
        p.ChangeDutyCycle(pctg)
        
    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()
        time.sleep(2)

"""
Capture camera images for prediction
"""
def camera_capture(camera):
    try:
        with picamera.array.PiYUVArray(camera) as stream:
            camera.capture(stream, 'yuv', use_video_port=True)
            image = stream.array.reshape((1, stream.array.shape[0], 
            stream.array.shape[1], stream.array.shape[2]))
            return image
    except KeyboardInterrupt:
        time.sleep(2)
        pass

"""
Autonomous steering prediction and Throttle control main loop
"""
def auto_mode_loop(my_cam,thr_p,pwm_pin,freq, steering):
    global ledState
    global count
    time.sleep(0.5)
    try:
        while (ledState == True):
            t0 = time.time()
            image = camera_capture(my_cam)
            t1 = time.time()            
            input_data = np.array(image, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            steer_val = interpreter.get_tensor(output_details[0]['index'])
            t2 = time.time()
            if (abs(steer_val[0][0]) > 0.3):
                thr_p.hardware_PWM(pwm_pin,freq, THR_SLOW *100)
            else:
                thr_p.hardware_PWM(pwm_pin,freq, THR_FAST *100)
            t3 = time.time()
            auto_steer(steer_val[0][0], steering)
            t4 = time.time()            
            print("Steering angle is {} and predict is : {}".format(conv.pwm_to_angle(
                         conv.norm_angle_to_pwm(steer_val[0][0])),steer_val[0][0]))

            if GPIO.input(10) == True:
                GPIO.output(LEDPin,GPIO.LOW)
                GPIO.output(LEDPin_2,GPIO.HIGH)
                ledState = False
                thr_p.hardware_PWM(pwm_pin,freq, THR_IDLE *100) 
                print("Autonomous driving session terminated by button press !")
                time.sleep(2)
                count +=1
                print('Remaining autonomous driving sessions : {}'.format(blinkCount - count))

                if count >= blinkCount:
                    GPIO.output(LEDPin_2,GPIO.LOW)
                    print("Autonomous driving sessions completed !")
                    GPIO.cleanup()
                    thr_p.hardware_PWM(pwm_pin,freq, THR_IDLE *100) 
                    steering.stop()
                    time.sleep(1)
                    ledState = False 
                    pass
                else:
                    print("Press button to continue Autonomous Driving")
                    pass

    except KeyboardInterrupt:
        time.sleep(1)
        GPIO.cleanup()
        thr_p.hardware_PWM(pwm_pin,freq, THR_IDLE *100)
        steering.stop()
        GPIO.output(LEDPin,GPIO.LOW)
        GPIO.output(LEDPin_2,GPIO.HIGH)
        ledState = False        
        print("\nAutonomous driving terminated by user !")
        pass

##### MAIN ###########
if __name__ == '__main__':
    
    # pin initializations
    blinkCount = 1
    count = 0 
    buttonPin = 10 
    LEDPin = 16
    LEDPin_2 = 18
    servoPIN = 11
    freq = 60
    pwm_pin = 19 
    cal_done = 0

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="Lite_Models/e40_spe6k_bs64_split0.2/model-ds1-021-006748.tflite")
    #interpreter = tf.lite.Interpreter(model_path="Lite_Models/model-ds1-040-0.002877.tflite")
    #interpreter = tf.lite.Interpreter(model_path="Models/converted_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']

    # initialize steering and camera 
    steering = init_auto_steer(LEDPin,LEDPin_2,buttonPin,servoPIN)
    thr_p = throttle_init(pwm_pin)
    cal_done = throttle_calibration(thr_p,pwm_pin,freq)

    if cal_done == 1 :
        print("\n*****************")
        print("Autonomous Driving")
        print("*****************")
        my_cam = cam_init() 
        GPIO.setup(buttonPin, GPIO.IN, pull_up_down = GPIO.PUD_DOWN) 
        buttonPress = True
        ledState = False
        try:
            GPIO.output(LEDPin,GPIO.LOW)
            GPIO.output(LEDPin_2,GPIO.HIGH)
            print("Press button to start Autonomous Driving")
            while count < blinkCount:
                buttonPress = GPIO.input(buttonPin)
                if buttonPress == True and ledState == False:
                    GPIO.output(LEDPin,GPIO.HIGH)
                    GPIO.output(LEDPin_2,GPIO.LOW)
                    ledState = True
                    print("Autonomous Driving mode enabled")
                    auto_mode_loop(my_cam,thr_p,pwm_pin,freq, steering)

        except KeyboardInterrupt:
            time.sleep(2)
            GPIO.cleanup()
            thr_p.hardware_PWM(pwm_pin,freq, THR_IDLE *100)
            steering.stop()
            time.sleep(1)
            ledState = False 
            print("\nAutonomous driving terminated by user !")
            pass
