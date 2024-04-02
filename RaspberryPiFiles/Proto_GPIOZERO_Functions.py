from gpiozero import DistanceSensor, AngularServo, RGBLED, DigitalOutputDevice
import time


################################################# CONSTANT DECLARATIONS

############################# GPIO Pin Declarations
RED_PIN = 1
GREEN_PIN = 7
BLUE_PIN = 8
TRIG_PIN = 20
ECHO_PIN = 21
FRONT_SERVO_PIN = 12
BACK_SERVO_PIN = 13
SOLENOID_EN_PIN = 0
SOLENOID_EXTEND_PIN = 5
SOLENOID_RETRACT_PIN = 6

############################# Mathematical Constants
US_TO_DUTY = 0.037037

############################# Timing Constants
TIME_SLEEP_DIST_SENS = 0.5
TIME_NO_MOVEMENT = 5
TIME_BEFORE_DIST_SENS = 3
TIME_REAR_CHUTE_OPEN = 10

############################# Servo Angle Constants
FRONT_OPEN_ANGLE = 60
FRONT_CLOSE_ANGLE = 0
BACK_OPEN_ANGLE = 135
BACK_CLOSE_ANGLE = 0

############################# LED Color Constants
CHUTE_LED_COLOR = (0, 0, 1)  # Blue
DOOR_LED_COLOR = (0, 1, 0)   # Green


################################################ GPIO INITIALIZATION

#Updated for RPI 5
ultrasonic = DistanceSensor(trigger = TRIG_PIN, echo = ECHO_PIN)
frontServo = AngularServo(FRONT_SERVO_PIN, min_angle=-135, max_angle=135)
backServo = AngularServo(BACK_SERVO_PIN, min_angle=-135, max_angle=135)
led = RGBLED(RED_PIN, GREEN_PIN, BLUE_PIN)
extend = DigitalOutputDevice(SOLENOID_EXTEND_PIN, initial_value(False))
retract = DigitalOutputDevice(SOLENOID_RETRACT_PIN, initial_value(False))


###
"""
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.setup(FRONT_SERVO_PIN, GPIO.OUT)
GPIO.setup(BACK_SERVO_PIN, GPIO.OUT)
GPIO.setup(SOLENOID_EXTEND_PIN, GPIO.OUT)
GPIO.setup(SOLENOID_RETRACT_PIN, GPIO.OUT)
"""


################################################ SERVO FUNCTIONS

"""def servo_write(pwm_obj, angle):
	"
	Calculates the PWM duty cycle to send to the servo motor given
	an angle (degrees) 
	"
	for servo_angle in range(0,280,10):
		pwm_val = (servo_angle*US_TO_DUTY)+2.5
		pwm_obj.ChangeDutyCycle(pwm_val)
		print(f"Angle: {servo_angle} | PWM: {pwm_val}")
"""

def front_servo_open():
	"""
	Controls the speed at which the front servo motor is able to open
	"""
	frontServo.angle = FRONT_OPEN_ANGLE
	# implement some function to control speed servo opens and closes

def front_servo_close():
	"""
	Controls the speed at which the front servo motor is able to close
	"""
	frontServo.angle = FRONT_CLOSE_ANGLE
	# functon to control speed hre

def back_servo_open():
	"""
	Controls the speed at which the back servo is able to open
	"""
	backServo.angle = BACK_OPEN_ANGLE
	# function here

def back_servo_close():
	"""
	Controls the speed at which the back servo is able to close
	"""
	backServo.angle = BACK_CLOSE_ANGLE
	#function ehre


################################################ SOLENOID FUNCTIONS

def solenoid_pulse(SOLENOID_PIN):
	"""
	Sends a pulse to the specified solenoid terminal to retract or
	extend the solenoid. Depending on the pulse that is sent, the
	lenght of the pulse will be set and the pulse will be sent
	to the solenoid and either extend or retract
	"""

	PULSE_TIME = 0.04

	# define length of pulse sent to the solenoid
	if SOLENOID_PIN == extend:
		PULSE_TIME = 0.04
	elif SOLENOID_PIN == retract:
		PULSE_TIME = 0.07

	SOLENOID_PIN.on()
	time.sleep(PULSE_TIME)
	SOLENOID_PIN.off()

def solenoid_extend():
	"""
	Extends the solenoid, putting the door into the locked postion
	"""
	solenoid_pulse(extend)

def solenoid_retract():
	"""
	Retracts the solenoid, putting the door into the unlocked position
	"""
	solenoid_pulse(retract)


############################################### LED FUNCTIONS

def led_blink_on(color):
	"""
	Begins in an OFF state, blinks the LED three times, then
	leaves the LED ON until ledBlinkOff is ran
	"""
	blink_led = led

	if color=="green":
		blink_led.color = (0, 1, 0)
	elif color=="blue":
		blink_led.color = (0, 0, 1)

	# Blink code
	blink_led.pulse(on_color(blink_led.color), off_color(0, 0, 0), n=3)
	"""
    for i in range(0, 3):
        blink_led.on()
        time.sleep(0.7)
        blink_led.off()
        time.sleep(0.7)
    """

def led_blink_off(color):
	"""
	Begins in an ON state, blinks the LED three times, then
	laeves the LED OFF until ledBlinkOn is ran
	"""
	blink_led = led
	
	if color=="green":
		ledcolor = blink_led.color = (0, 1, 0)
	elif color=="blue":
		ledcolor = blink_led.color = (0, 0, 1)

	# Blink code
	blink_led.pulse(on_color(ledcolor), off_color(0, 0, 0), n=3)


################################################ ULTRASONIC FUNCTIONS
"""
def get_distance():
	""
	Sends an ultrasonic trigger, listens for the echo, & calculates the distance.
	Returns distance.
	""
	# send pulse to trigger pin to send ultrasound wave
	ultrasonic.trigger()
	print("sent trigger")
	# start recording time & wait for ultrasound to return
	start_time = time.time()
	stop_time = time.time()

	print("waiting for echo")
	while ultrasonic.distance == 0:
        	start_time = time.time()
    	while ultrasonic.distance > 0:
        	stop_time = time.time()

    	print("calculating distance")
    	# calculate time elapsed, then multiply by speed of sound
    	dist = (stop_time - start_time) * 34300 / 2  # divide by 2 since pulse travels there and back
    	return dist
"""
def sense_movement():
	"""
	Senses for movement inside of the package chute to automatically close the chute. Holds the 
	program flow in a while loop until either max time elapsed or no objects are detected 
	inside of the package chute.
	"""

	dist_curr = round((ultrasonic.distance) * 100, 4)
    	dist_prev = dist_curr

    	start_time = time.time()
    	timer = time.time()

    	movement_sensed = True
    	closing_timer = 0
    	present_timer = 0

    	while (present_timer - closing_timer < TIME_NO_MOVEMENT):
        	time.sleep(TIME_SLEEP_DIST_SENS)
        	dist_prev = dist_curr  # set prev distance to current
        	dist_curr = round((ultrasonic.distance) * 100, 4)
        	print(f"current dist: {dist_curr}\tprevious dist: {dist_prev}")
        	movement_boolean = (dist_curr < dist_prev + 4 and dist_prev - 4 < dist_curr)

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

			print(f"movement detected in chute reseting timers..")

################################################ PACKAGE CHUTE FUNCTIONS

def activate_chute():
	"""
	Activates the servo chute through controlling the PWM sent to the servo motor
	and sensing movement inside the package chute to determine when the chute should
	be closed. This function is driven based on classifications from the object
	detection neural network
	"""
	print("CHUTE ACTIVATED")

	led_blink_on(CHUTE_LED_COLOR)

	front_servo_open()
	time.sleep(TIME_BEFORE_DIST_SENS)

	sense_movement()

	led_blink_off(CHUTE_LED_COLOR)

	front_servo_close()
	time.sleep(1)

	back_servo_open()
	time.sleep(TIME_REAR_CHUTE_OPEN)
	back_servo_close()

	print("CHUTE OPERATION COMPLETE")


################################################ DOOR LOCKING FUNCTIONS

def unlock_door():
	"""
	Unlocks the door. First, the LED is enabled green, and the solenoid is
	retracted
	"""

	print("DOOR UNLOCKING")

	led_blink_on(DOOR_LED_COLOR)

	time.sleep(0.3)

	solenoid_retract()

def lock_door():
	"""
	Locks the door. First, the LED is disabled from green, and the solenoid
	is extended
	"""

	print("DOOR LOCKED")

	led_blink_off(DOOR_LED_COLOR)

	time.sleep(0.3)

	solenoid_extend()
