# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 00:43:56 2020

@author: Naveen Chengappa
"""

MAX_PWM_READ = 1830
MIN_PWM_READ = 1140

# MAX_PWM_READ = 1848
# MIN_PWM_READ = 1147

ANGLE_LEFT = -30
ANGLE_RIGHT = 30

NORMALIZE_ANGLE_L = -1
NORMALIZE_ANGLE_R = 1

Difference = MAX_PWM_READ - MIN_PWM_READ
pwm = 0

def pwm_to_bits(pwm):
    bits = ((pwm - MIN_PWM_READ)*255)/Difference
    return int(bits)
 
def pwm_to_angle(pwm):
    if pwm > MAX_PWM_READ:
        angle = ANGLE_RIGHT
    elif pwm < MIN_PWM_READ:
        angle = ANGLE_LEFT
    else:
        angle = (((ANGLE_RIGHT - ANGLE_LEFT)*(pwm - MIN_PWM_READ)) / Difference) + ANGLE_LEFT        
    return round(angle,2)
    
def pwm_to_norm_angle(pwm):
    if pwm > MAX_PWM_READ:
        angle = NORMALIZE_ANGLE_R
    elif pwm < MIN_PWM_READ:
        angle = NORMALIZE_ANGLE_L
    else:
        angle = (((NORMALIZE_ANGLE_R - NORMALIZE_ANGLE_L)*(pwm - MIN_PWM_READ)) / Difference) + NORMALIZE_ANGLE_L        
    return round(angle, 2)

def bits_to_pwm(bits):
    pwm = ((Difference * bits) / 255) + MIN_PWM_READ
    return float(pwm)

def angle_to_pwm(angle):
    pwm = (((angle - ANGLE_LEFT)*Difference) / (ANGLE_RIGHT - ANGLE_LEFT)) + MIN_PWM_READ
    return float(pwm)

def norm_angle_to_pwm(angle):
    pwm = (((angle - NORMALIZE_ANGLE_L)*Difference) / (NORMALIZE_ANGLE_R - NORMALIZE_ANGLE_L)) + MIN_PWM_READ
    return float(pwm)
    
def angle_to_bits(angle):
    bits = 255 * ((angle - ANGLE_LEFT) / (ANGLE_RIGHT - ANGLE_LEFT))
    return int(bits)

def bits_to_angle(bits):
    angle = ANGLE_LEFT + ((bits * (ANGLE_RIGHT - ANGLE_LEFT))/255)
    return float(angle)

