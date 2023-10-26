import RPi.GPIO as GPIO
import time
import gpio_functions as GPIO_fun

LOCK_PIN = 00 # pin 37 bcm 26
CHUTE_PIN = 26 # pin 37 bcm 26
TRIG_PIN = 18 # pin 12 bcm 18
ECHO_PIN = 12 # pin 32 bcm 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)
GPIO.setup(CHUTE_PIN, GPIO.OUT)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.output(TRIG_PIN, GPIO.LOW)
GPIO.setup(ECHO_PIN, GPIO.IN)
time.sleep(1)

chute = GPIO.PWM(CHUTE_PIN, 50) # GPIO 17 for PWM with 50Hz
chute.start(2.5) # Initialization to 0

try:
	while True:
		GPIO_fun.servo_write(0)
		time.sleep(0.5)
		GPIO_fun.servo_write(60)
		time.sleep(5)
		print("sensing movement...")
		GPIO_fun.sense_movement()
		GPIO_fun.servo_write(0)
		print("finished")

except KeyboardInterrupt:
	chute.stop()
	GPIO.cleanup()