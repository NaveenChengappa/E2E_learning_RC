
import time
import pigpio

pin = 19
p = pigpio.pi() # Connect to local Pi.
freq = 60
p.set_mode(pin,pigpio.OUTPUT)
p.hardware_PWM(19,freq, 74500)

#p.set_servo_pulsewidth(19, 1600)

while True :
	#p.hardware_PWM(19,freq, 77490)
	p.hardware_PWM(19,freq, 940 *100)
	# time.sleep(0.5)

# p.hardware_PWM(19,freq,0 *100)
# p.stop()

# pi.hardware_PWM(19,800,1e6*0.25)
# pi.set_PWM_dutycycle(19, 255*0.25)

# pi.set_PWM_frequency(pin,50)
# pi.set_PWM_range(pin,20000)

# pi.set_servo_pulsewidth(12, 1500)
# time.sleep(5)
# # switch servo off
# pi.set_servo_pulsewidth(12, 0);

# pi.stop()
