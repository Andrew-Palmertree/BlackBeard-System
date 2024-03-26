from time import sleep
import Proto_GPIOZERO_Functions as PROTO_GPIO

# User Interface Loops
while True:
	num_in = str(input("Enter a number:\n1. OPERATE SERVO CHUTE\n2.OPERATE SOLENOID\n"))
	if num_in=="1":
		print("Servo Operate chosen")
		PROTO_GPIO.activate_chute(frontServo, backServo)
	elif num_in=="2":
		print("Solenoid Operate  chosen")
		PROTO_GPIO.unlock_door()
		sleep(3)
		PROTO_GPIO.lock_door()
	else:
		print("INVALID INPUT ENTERED")

