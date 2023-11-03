import numpy as np
import cv2
import os
from datetime import datetime
import time

# Prompt the user for a name to add to the list of known users
user_input = input("Please enter name to add to the known users:\n")
print("\nYou entered:", user_input, "\n")

# Define the base directory and initialize counters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_counter = 0  # Counter to keep track of the number of images saved
number_images = 30 # Counter of max images when continuously taking photos of full images

# OpenCV's Haar Cascade classifier for face detection
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# Create the directory structure if they don't exist
save_dir_root = os.path.join(os.getcwd(), user_input) # Current_dir_the_file_in/user_input
save_dir_images = os.path.join(os.getcwd(), user_input + f"/{user_input}_images") # Current_dir_the_file_in/user_input/user_input_images
cascade_dir_images = os.path.join(os.getcwd(), user_input + f"/{user_input}_cascade_images") # Current_dir_the_file_in/user_input/user_input_cascade_images
cropped_dir = os.path.join(os.getcwd(), user_input + f"/{user_input}_cropped")# Current_dir_the_file_in/user_input/user_input_cropped

# Create the (Current_dir_the_file_in/user_input) directory if it doesn't exist (user_input)
if not os.path.exists(save_dir_root):
    os.makedirs(save_dir_root)

# Create the (Current_dir_the_file_in/user_input/user_input_images) directory if it doesn't exist (user_images)
if not os.path.exists(save_dir_images):
    os.makedirs(save_dir_images)

# Create the (Current_dir_the_file_in/user_input/user_input_cascade_images) directory if it doesn't exist (user_input_cascade_images)
if not os.path.exists(cascade_dir_images):
    os.makedirs(cascade_dir_images)
    
# Create the (Current_dir_the_file_in/user_input/user_input_cropped) directory if it doesn't exist (user_input_cropped)
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)    

# Function to extract a face from an image
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face is detected, it returns None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None  # No faces detected
    else:
        (x, y, w, h) = faces[0]
        cropped_face = img[y:y + h, x:x + w]
        return cropped_face   

# Initialize the camera capture    
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

recognition_mode = False  # Flag to control face recognition mode
count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(20)

# Will capture a single image and save that image into the user_input/user_input_images folder    
    if key & 0xFF == ord('s'):
        # Saving a single image
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        image_filename = f"{user_input}_{current_time}.jpg"
        image_path = os.path.join(save_dir_images, image_filename)
        cv2.imwrite(image_path, frame)
        print(f'\nImage saved as {image_filename}')
        time.sleep(0.5)

# Will continuously capture images until the desired number of images, as set by the 'number_images' variable, is reached        
    if key & 0xFF == ord('c'):
        # Saving number_images photos every one second
        for images in range(number_images):
            ret, frame = cap.read()
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            image_filename = f"{user_input}_{current_time}.jpg"
            image_path = os.path.join(save_dir_images, image_filename)
            cv2.imwrite(image_path, frame)
            print(f'\nImage saved as {image_filename}')
            cv2.imshow('frame', frame)
            cv2.waitKey(1000)
            image_counter += 1
            if image_counter >= number_images:
                break
                
# This is the most efficient way to capture and save 300x300 cropped images
# Images are continuously cropped and saved to a folder named after the user's input (user_input/user_input_cascade_images)
    if key & 0xFF == ord('r'):
        if recognition_mode:
            recognition_mode = False  # Turn off recognition mode
        else:
            recognition_mode = True  # Turn on recognition mode
            count = 0  # Reset the count when turning on recognition mode

    if recognition_mode:
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (300, 300))
            file_name_path = os.path.join(cascade_dir_images, f"cascade_images" + str(count) + ".jpg")
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

