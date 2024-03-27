import RPi.GPIO as GPIO
import time

def get_distance():
	# send pulse to trigger pin to send ultrasound wave
	GPIO.output(trigger_pin,GPIO.HIGH)
	time.sleep(0.00001)
	GPIO.output(trigger_pin,GPIO.LOW)
	print("sent trigger")
	# start recording time & wait for ultrasound to return
	start_time = time.time()
	stop_time = time.time()

	print("waiting for echo")
	while GPIO.input(echo_pin) == 0:
		start_time = time.time()
	while GPIO.input(echo_pin) == 1:
		stop_time = time.time()

	print("calculating distance")
	# calculate time elapsed, then multiply by speed of sound
	dist = (stop_time - start_time) * 34300 / 2 # divide by 2 since pulse travels there and back
	return dist

def sense_movement():
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

def servo_write(angle):
	# pwm ranges (0-10) + 2.5 --> 2.5-12.5
	# angle ranges 0-180
	pwm = (angle/18) + 2.5
	p.ChangeDutyCycle(pwm)
	print(f"angle: {angle}\tpwm: {pwm}")

servo_pin = 26 # pin 37 bcm 26
trigger_pin = 18 # pin 12 bcm 18
echo_pin = 12 # pin 32 bcm 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)
GPIO.output(trigger_pin, GPIO.LOW)
time.sleep(1)

p = GPIO.PWM(servo_pin, 50) # GPIO 17 for PWM with 50Hz
p.start(2.5) # Initialization to 0
try:
	while True:
		servo_write(0)
		time.sleep(0.5)
		servo_write(60)
		time.sleep(5)
		print("sensing movement...")
		sense_movement()
		servo_write(0)
		print("finished")

except KeyboardInterrupt:
	p.stop()
	GPIO.cleanup()
