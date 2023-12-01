import RPi.GPIO as GPIO
import time
import gpio_functions as GPIO_func

# Initialize GPIO pins
LOCK_PIN = 5 # pin x bcm 5
CHUTE_PIN = 22 # pin 37 bcm 22
TRIG_PIN = 18 # pin 12 bcm 18
ECHO_PIN = 12 # pin 32 bcm 12
GLED_PIN = 13 # pin x bcm 13
BLED_PIN = 26 # pin x bcm 26
CBUTTON_PIN = 20 # pin x bcm 20
LBUTTON_PIN = 21 # pin x bcm 21

GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)
GPIO.setup(CHUTE_PIN, GPIO.OUT)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.output(TRIG_PIN, GPIO.LOW)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.setup(GLED_PIN, GPIO.OUT)
GPIO.setup(BLED_PIN, GPIO.OUT)
GPIO.setup(CBUTTON_PIN, GPIO.IN)
GPIO.setup(LBUTTON_PIN, GPIO.IN)

time.sleep(1)

# Setup both servo motors
chute = GPIO.PWM(CHUTE_PIN, 50) # GPIO 17 for PWM with 50Hz
chute.start(2.5) # Initialization to 0

lock = GPIO.PWM(LOCK_PIN, 50)
lock.start(2.5)

GPIO.output(GLED_PIN, GPIO.LOW)
GPIO.output(BLED_PIN, GPIO.LOW)

def chute_activation():
	time.sleep(0.2)
	GPIO_func.open_chute(chute)
	time.sleep(5)
	print("sensing movement...")
	GPIO_func.sense_movement()
	GPIO_func.close_chute(chute)
	print("finished")

def lock_activation():
	time.sleep(0.2)
	print("locking door")
	GPIO_func.unlock_door(lock)
	time.sleep(3)
	print("locking door")
	GPIO_func.lock_door(lock)
	print("finished")

try:
	while True:
		# check if buttons are pressed, then do some action (open chute or unluck)
		chute_active = GPIO.input(CBUTTON_PIN)
		lock_active = GPIO.input(LBUTTON_PIN)
		print(f"CHUTE = {chute_active} | LOCK = {lock_active}")

		if not chute_active:
			chute_activation()
		elif not lock_active:
			lock_activation()

except KeyboardInterrupt:
	chute.stop()
	GPIO.cleanup()
