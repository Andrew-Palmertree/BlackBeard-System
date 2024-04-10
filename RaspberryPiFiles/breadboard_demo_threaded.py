from time import sleep
import Proto_GPIOZERO_Threaded_Functions as PROTO_GPIO

# User Interface Loops
while True:
	num_in = str(input("Enter a number:\n1. OPERATE SERVO CHUTE\n2.OPERATE SOLENOID\n"))
	if num_in=="1":
		print("Servo Operate chosen")
		#PROTO_GPIO.activate_chute()
		activate_chute = PROTO_GPIO.create_chute_thread()
		activate_chute.start()

	elif num_in=="2":
		print("Solenoid Operate  chosen")
		#PROTO_GPIO.unlock_door()
		#sleep(3)
		#PROTO_GPIO.lock_door()
		activate_door = PROTO_GPIO.create_door_thread()
		activate_door.start()

	else:
		print("INVALID INPUT ENTERED")

