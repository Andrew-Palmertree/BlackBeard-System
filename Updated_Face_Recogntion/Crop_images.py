import os
import numpy as np
import re 
# import dlib
import cv2
import pickle
from PIL import Image

# Import LBPHFaceRecognizer from cv2.face (contrib)
from cv2.face import LBPHFaceRecognizer


# The base dir. For example, /home/andrew/facial_recognition. Get the dir path from this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the directory path where you want to crop the faces 
dir_path = os.path.join(BASE_DIR, "pubfig/train/Hannah")

#cascade classifier
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# Initialize LBPHFaceRecognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Initialize a list to store the labels of the faces detected
count = 0

def face_extractor(img, path):
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


# Traverse through all files in the specified directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            
            print(path)
            # Load the image
            image = cv2.imread(path)
            
            
            # Extract and resize the detected face (if any)
            cropped_face = face_extractor(image, path)
            if cropped_face is not None:
                # Resize and save the face only when a face is detected
                face = cv2.resize(cropped_face, (300, 300))
                cv2.imwrite(path, face)
                count += 1
            else:
                # try and delete the images where faces are not recognized
                print(f"No face detected in {path}, deleting...")
                try:
                    os.remove(path)
                    print(f"Deleted: {path}")
                except OSError as e:
                    print(f"Error deleting {path}: {e}")

print("Count of images cropped and saved: " + str(count))
