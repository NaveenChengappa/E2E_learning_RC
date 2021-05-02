# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 00:43:56 2020

@brief : Parameter converter code
@author: Naveen Chengappa
"""

# Macro initializations
"""
Steering PWM reads from ADC
Max read --  full right steer (+30 degrees)
Min read --  full left steer (-30 degrees)
"""
MAX_PWM_READ = 1830 
MIN_PWM_READ = 1140
Difference = MAX_PWM_READ - MIN_PWM_READ

"""
Max angle while steering
"""
ANGLE_LEFT = -30.0
ANGLE_RIGHT = 30.0

"""
Normalized angles
-1 indicates full left steer
+1 indicates full right steer
"""
NORMALIZE_ANGLE_L = -1
NORMALIZE_ANGLE_R = 1

pwm = 0

def pwm_to_bits(pwm):
    """
    Brief      :: Converts PWM values to bits    
    Parameters :: pwm : Input PWM value within min-max range 
    Returns    :: bits : Integer in the range 0 - 255
    """
    bits = ((pwm - MIN_PWM_READ)*255)/Difference
    return int(bits)
 
def pwm_to_angle(pwm):
    """
    Brief      :: Converts PWM values to angles    
    Parameters :: pwm : Input PWM value within min-max range 
    Returns    :: angles : floating value in the range 30 to -30 degrees
    """
    if pwm > MAX_PWM_READ:
        angle = ANGLE_RIGHT
    elif pwm < MIN_PWM_READ:
        angle = ANGLE_LEFT
    else:
        angle = (((ANGLE_RIGHT - ANGLE_LEFT)*(pwm - MIN_PWM_READ)) / Difference) + ANGLE_LEFT        
    return round(angle, 2)

def pwm_to_norm_angle(pwm):
    """
    Brief      :: Converts PWM values to normalized angles    
    Parameters :: pwm : Input PWM value within min-max range 
    Returns    :: angles : floating value in the range 1 to -1 degrees
    """
    if pwm > MAX_PWM_READ:
        angle = NORMALIZE_ANGLE_R
    elif pwm < MIN_PWM_READ:
        angle = NORMALIZE_ANGLE_L
    else:
        angle = (((NORMALIZE_ANGLE_R - NORMALIZE_ANGLE_L)*(pwm - MIN_PWM_READ)) / Difference) + NORMALIZE_ANGLE_L        
    return round(angle, 2)

def bits_to_pwm(bits):
    """
    Brief      :: Converts bits to pwm values      
    Parameters :: bits : Integer in the range 0 - 255
    Returns    :: pwm : PWM value within min-max range 
    """
    pwm = ((Difference * bits) / 255) + MIN_PWM_READ
    return float(pwm)

def angle_to_pwm(angle):
    """
    Brief      :: Converts angles to pwm values      
    Parameters :: angle : floating value in the range 30 to -30 degrees
    Returns    :: pwm : PWM value within min-max range 
    """
    pwm = (((angle - ANGLE_LEFT)*Difference) / (ANGLE_RIGHT - ANGLE_LEFT)) + MIN_PWM_READ
    return float(pwm)

def norm_angle_to_pwm(angle):
    """
    Brief      :: Converts normalized angles to pwm values      
    Parameters :: angle : floating value in the range 1 to -1 degrees
    Returns    :: pwm : PWM value within min-max range 
    """
    pwm = (((angle - NORMALIZE_ANGLE_L)*Difference) / (NORMALIZE_ANGLE_R - NORMALIZE_ANGLE_L)) + MIN_PWM_READ
    return float(pwm)
    
def angle_to_bits(angle):
    """
    Brief      :: Converts angle values to bits    
    Parameters :: angle : floating value in the range 30 to -30 degrees
    Returns    :: bits : Integer in the range 0 - 255
    """
    bits = 255 * ((angle - ANGLE_LEFT) / (ANGLE_RIGHT - ANGLE_LEFT))
    return int(bits)

def bits_to_angle(bits):
    """
    Brief      :: Converts bits to angles   
    Parameters :: bits : Integer in the range 0 - 255
    Returns    :: angles : floating value in the range 30 to -30 degrees
    """
    angle = ANGLE_LEFT + ((bits * (ANGLE_RIGHT - ANGLE_LEFT))/255)
    return float(angle)

