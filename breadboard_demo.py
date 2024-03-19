from time import sleep
import RPi.GPIO as GPIO
import Proto_GPIO_Functions as PROTO_GPIO

############################################## 	PWM OBJECT INIT
front_servo = GPIO.PWM(12, 50)
front_servo.start(0)

back_servo = GPIO.PWM(13, 50)
back_servo.start(0)

# User Interface Loops
while True:
	num_in = str(input("Enter a number:\n1. OPERATE SERVO CHUTE\n2.OPERATE SOLENOID\n"))
	if num_in=="1":
		print("Servo Operate chosen")
		PROTO_GPIO.activate_chute(front_servo, back_servo)
	elif num_in=="2":
		print("Solenoid Operate  chosen")
		PROTO_GPIO.unlock_door()
		sleep(3)
		PROTO_GPIO.lock_door()
	else:
		print("INVALID INPUT ENTERED")

