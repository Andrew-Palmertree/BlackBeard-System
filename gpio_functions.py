import RPi.GPIO as GPIO
import time


# Initialize GPIO pins
LOCK_PIN = 5 # pin x bcm 5
CHUTE_PIN = 22 # pin 37 bcm 22
TRIG_PIN = 18 # pin 12 bcm 18
ECHO_PIN = 12 # pin 32 bcm 12
GLED_PIN = 13 # pin x bcm 13
BLED_PIN = 26 # pin x bcm 26
CBUTTON_PIN = 20 # pin x bcm 20
LBUTTON_PIN = 21 # pin x bcm 21


def get_distance():
	"""
	Sends an ultrasonic trigger, listens for the echo, & calculates the distance.
	Returns distance.
	"""

	# send pulse to trigger pin to send ultrasound wave
	GPIO.output(TRIG_PIN,GPIO.HIGH)
	time.sleep(0.00001)
	GPIO.output(TRIG_PIN,GPIO.LOW)
	print("sent trigger")
	# start recording time & wait for ultrasound to return
	start_time = time.time()
	stop_time = time.time()

	print("waiting for echo")
	while GPIO.input(ECHO_PIN) == 0:
		start_time = time.time()
	while GPIO.input(ECHO_PIN) == 1:
		stop_time = time.time()

	print("calculating distance")
	# calculate time elapsed, then multiply by speed of sound
	dist = (stop_time - start_time) * 34300 / 2 # divide by 2 since pulse travels there and back
	return dist

def sense_movement():
	"""
	Senses for movement inside of the package chute to automatically close the chute. Holds the 
	program flow in a while loop until either max time elapsed or no objects are detected 
	inside of the package chute.
	"""

	dist_curr = get_distance()
	dist_prev = dist_curr

	start_time = time.time()
	timer = time.time()

	# TODO: when no movement is sensed for 5 seconds, then close the box
	# current function: closes when 5 seconds elapsed or movement detected

	movement_sensed = True
	closing_timer = 0
	present_timer = 0
	no_movement_time = 5 # time to wait for no movement in package chute before closing

	while (present_timer-closing_timer<no_movement_time):
		time.sleep(0.5)
		dist_prev = dist_curr # set prev distance to current
		dist_curr = get_distance()
		print(f"current dist: {dist_curr}\tprevois dist: {dist_prev}")
		movement_boolean = (dist_curr<dist_prev+4 and dist_prev-4<dist_curr)

		# analyze different scenarios while sensing for pack
		if movement_boolean and movement_sensed:
			# no movement detected & movement previously detected
			# start logging time that there is no movement inside chute
			closing_timer = time.time()
			present_timer = closing_timer
			movement_sensed = False

			print(f"no movement sensed! starting timing for closing..")
		elif movement_boolean and not movement_sensed:
			# no movement detected & no movement previously detected
			# continue polling, see if no movement detected for 5s
			present_timer = time.time()

			print(f"no movement sensed, time = {present_timer - closing_timer}")
		elif not movement_boolean and movement_sensed:
			# movement detected & movement previously detected
			# ensure timers still set at 0
			closing_timer = 0
			present_timer = 0

			print(f"movement detected in chute still")
		elif not movement_boolean and not movement_sensed:
			# movement detected & movement not previosly detected
			# reset timers & change movement_sensed variable
			closing_timer = 0
			present_timer = 0
			movement_sensed = True

			print(f"movement detected in chute1 reseting timers..")

def servo_write(angle,pwm_obj):
	"""
	Changes the angle of the servo motor. Desired angle passed in as argument
	"""

	# pwm ranges (0-10) + 2.5 --> 2.5-12.5
	# angle ranges 0-180
	pwm = (angle/18) + 2.5
	pwm_obj.ChangeDutyCycle(pwm)
	print(f"angle: {angle}\tpwm: {pwm}")

def led_blink(led_pin):
	"""
	Blinks the LED pin passed into function
	"""
	for i in range(3):
		GPIO.output(led_pin, GPIO.LOW)
		time.sleep(0.5)
		GPIO.output(led_pin, GPIO.HIGH)
		time.sleep(0.5)

def open_chute(pwm_obj):
	""" 
 	Opens the package chute door 
  	"""
	led_blink(BLED_PIN)
	GPIO.output(BLED_PIN, 1)

	for i in range(0,62,2):
                servo_write(i,pwm_obj)
                time.sleep(0.1)

def close_chute(pwm_obj):
	""" 
 	Closes the package chute door
  	"""
	led_blink(BLED_PIN)
	GPIO.output(BLED_PIN, 0)

	for i in range(60,-2,-2):
                servo_write(i,pwm_obj)
                time.sleep(0.1)

def unlock_door(pwm_obj):
	""" 
 	Unlocks the deadbolt on the door
  	"""
	servo_write(135,pwm_obj)

	led_blink(GLED_PIN)
	GPIO.output(GLED_PIN, 1)

def lock_door(pwm_obj):
	""" 
 	Locks the deadbolt on the door
  	"""
	led_blink(GLED_PIN)
	GPIO.output(GLED_PIN, 0)

	servo_write(0,pwm_obj)

