#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: main_face_rec.py
# Author of the script: Andrew Palmertree
# Date: March 9, 2024
# Version: 1.0.0
# 
# Requirments:
#     - python 3.9.18
#     - check the requirement.txt file for specific packages
#
# Usage: 
#     - used for the AppServer2 script
#     - can be used by itself in a command line to get the image count for each user class in pubfig/train: 
#       python converter2.py  
#
# Description:
#     The main_face_rec script is for the BlackBeard project to be used on the server side to extract detect faces from videos and images.
#     This script also creates the necessary folders for new user classes.
#     This script has the logic to get an image count of all user classes in a sepcific folder.
#
# Notes:
# This uses Deep Neurel Network (DNN) for the video face extract
# and uses openCV Haar cascade and MTCNN for image face extract
#
#############################################################################################


#============================================================================================
# Import the necessary package libraries
#============================================================================================
import os
import cv2
import numpy as np
import shutil
import random
import time
from mtcnn import MTCNN
from mtcnn.exceptions import InvalidImage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import tensorflow as tf
from cv2.face import LBPHFaceRecognizer


#============================================================================================
# Initialize variables 
#============================================================================================
## Assign folder locations
train_dir = 'pubfig/train/'
switchFolder = 'pubfig/switchFolder'

# Gets the absolute file path from where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File location where the cascade classifier is located 
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# Set the target number of images where the user classes cannot go over
target_images = 55

# Set new_dir as a global variable which will be used to assign and create new directories in multiple functions
new_dir = " "

# Initialize an empty list to store folder names with at least 45 images
folder_name_list = []

# Initialize an empty list to store all user classes
all_folder_name_list = []
            
# Keeps track of folders from pubfig/switchFolder
switch_Folders = []

# Keep track if an image has been cropped or not
cropped = False

# Flag to track if all directories have at least 45 images
all_have_45 = True

# Used to tell if the model has been trained with the new images and users
trained = False

# Used to tell if the model has been converted from keras (.h5) to tensorflow lite file (.tflite)
converted = False

# Used to tell if the model has been converted from tensorflow lite file (.tflite) to quantized to edgetpu tflite (edgetpu.tflite)
quantized = False

# Initialize LBPHFaceRecognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Initialize MTCNN face recognition
detector = MTCNN()

# Used to keep count of the faces in a video
face_count = 0

# Used to keep track if a video has been uplaoded and finished processing
videoUploading = 0

# determines if there's a detected face in an image or video frame
face_detected = False

# Load the pre-trained model and configuration for DNN face detection
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"    
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


#============================================================================================
# Define used functions
#============================================================================================


#############################################################################################
# Get the total count of images in the specified directory in temp/<name>
#############################################################################################
def getCountInDir(name):
    global face_count, detector
    
    # Specify the directory you want to use to count the images in
    directory_path = f"temp/{name}"

    # Specify the image file extensions you want to count
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    # Get a list of all files in the directory
    files_in_directory = os.listdir(directory_path)
    
    # Initilize MTCNN face detection
    detector = MTCNN()

    # Initialize a counter for image files
    face_count = 0

    # Loop through all the files
    for filename in files_in_directory:
        # Loop through all the image extensions
        for extension in image_extensions:
            # If the file ends with this extension, increment the counter
            if filename.endswith(extension):
                face_count += 1
    
    return face_count


#############################################################################################
# This function extracts faces using Haar Cascade method
# This method isn't used in images or videos 
# but can be switched out with another mehtod below
#############################################################################################
def Haar_Cascade_face_extraction(img, image_path):
    global face_count
    
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    # Return if no faces are detected
    if len(faces) == 0:
        print("No faces detected.")
        return None

    # Extract faces with Haar Cascade
    for results in faces:
        x, y, w, h = results['box']
        padding = 150  # Adjust this value to get the desired amount of room around the face
        x = max(0, x - padding)
        y = max(0, y - padding)
        w += padding * 2
        h += padding * 2
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = img[y:y+h, x:x+w]
        # Where the image will be saved
        image_path = os.path.join(temp_dir, 'face_{}.png'.format(face_count))

        cv2.imwrite(image_path, roi_color)
    

#############################################################################################
# This function extracts faces using the DNN method for images and videos
#############################################################################################    
def DNN_face_extraction(img, name):
    # Grab the temporary file location for the user class
    temp_dir = f"temp/{name}"
    
    # Get the height and width of the frame
    h, w = img.shape[:2]

    # Prepare the image for input to the DNN face extraction
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

    # Set the input blob for the neural network
    net.setInput(blob)

    # Perform forward pass inference to detect faces
    detections = net.forward()

    cropped_face = None

    # # Extract just the face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If the face extraction method has greater than %50 confidence there is a face, then it will extract the face
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Ensure the bounding box coordinates are within the image dimensions
            x = max(0, x)
            y = max(0, y)
            x1 = min(w, x1)
            y1 = min(h, y1)

            # Extract the face region
            roi_color = img[y:y1, x:x1]

            padding = 15  # Add padding so MTCNN can easily double check the faces
            roi_color_padded = cv2.copyMakeBorder(roi_color, padding, padding, padding, padding, \
                          cv2.BORDER_CONSTANT, value=[0, 0, 0])


            # Resize the face region to the desired size (96, 112)
            roi_color_resized = cv2.resize(roi_color, (96, 112))

            # Where the image will be saved
            image_path = os.path.join(temp_dir, 'face_{}.png'.format(face_count))

            # Save the face to a file in temp/<name> 
            cv2.imwrite(image_path, roi_color_resized)
          
            return image_path

        else: 
            image_path = None
                   
    return image_path


#############################################################################################
# This function extracts faces using the MTCNN method for images and videos
#############################################################################################
def MTCNN_face_extraction(image_path):
    
    face_detected = False
    
    # Function detects faces and returns the cropped face
    # If no face is detected, it returns None
    def face_extractor(img):  
        # Sees if there is a face in the image 
        try:
            faces = detector.detect_faces(img)
        except tf.errors.InvalidArgumentError as e:
            print(f"An error occurred during face detection: {e}")
            return None

        # if no faces are detected then returns None
        if faces == []:
            return None 
        else:
            x, y, w, h = faces[0]['box']
            cropped_face = img[y:y + h, x:x + w]
            return cropped_face

    # First reads the input image         
    image = cv2.imread(image_path)

    # Determines if there is a face in the image
    cropped_face = face_extractor(image)

    # If there is face in the imaage then crop it and save to temp/<name> file
    if cropped_face is not None:
        # Resize and save the face only when a face is detected
        faces = cv2.resize(cropped_face, (96, 112))
        # Save the image
        cv2.imwrite(image_path, faces)
        face_detected = True
        
    return face_detected
            

#############################################################################################
# This function extracts faces from uploaded videos 
#############################################################################################
def videoFaceExtraction(VIDEO_FILE, name):
    global face_count, detector, videoUploading, cropped, face_detected
    
    # Keep track if an image has been cropped or not
    cropped = True
    
    # Used to keep track if a video has been uplaoded and finished processing
    videoUploading = 1 
    
    # Open a video capture object to read from the specified video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    # Grabs the next available number in temp/<name> folder
    face_count = getCountInDir(name)
     
    # Initilize the face count and video frame count variables     
    count_images = 0
    frame_count = 0
    
    # Process the video to extract detected faces 
    while True:
        # Read the frame
        ret, img = cap.read()
        frame_count += 1  # Increment the frame counter
        
        # If there are no more frames, break the loop
        if not ret:
            print("No more frames to read")
            break

        # Only process every 4th frame
        if frame_count % 4 == 0:
            try:
                # Extracts faces first with DNN
                image_path = DNN_face_extraction(img, name)
                
                # Make sure that it detects a face using MTCNN
                if image_path != None:
                    face_detected = MTCNN_face_extraction(image_path)
                    
                # If a face is detected, then increment the count
                if (face_detected == True):
                    count_images += 1
                    face_count += 1
                    try:
                        # Display the image
                        cv2.imshow('img', img)
                    except cv2.error as e:
                        print("Video Finshed")
                        continue

            except cv2.error as e:
                print("Video Finshed")
                break

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        # press the 'Esc' key to exit the display
        if k==27:
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


#############################################################################################
# This function extracts detected faces from uploaded images
#############################################################################################
def cropImage(image_path, name):
    global cropped, face_detected

    # Detects any face in the image using MTCNN
    face_detected = MTCNN_face_extraction(image_path)

    # If a face is detected, then increment the count
    if (face_detected == True):
        cropped = True
    else:
        # try and delete the images where faces are not recognized
        print(f"No face detected in {image_path}, deleting...")
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except OSError as e:
            print(f"Error deleting {image_path}: {e}")            

            
#############################################################################################
# Functions to get the next available number that can be used from a specific directory
#############################################################################################           
def getNextAvailableNumber(directory):
    
    # Get the list of directories in the given directory
    directories = next(os.walk(directory))[1]

    # Extract the numbers from the directory names and sort them
    numbers = [int(dir_name.split('-')[0]) for dir_name in directories]
    numbers.sort()

    # Find the next missing number
    next_number = 1
    while next_number in numbers:
        next_number += 1

    return next_number


#############################################################################################
# This function is used to move folders from 'pubfig/train' to 'pubfig/switchFolder'
# This will reduce complexity of having to make dulplicate processes for both 
# 'pubfig/switchFolder' and 'pubfig/train'
#############################################################################################
def moveFolders():
    global train_dir, switchFolder

    # Get the list of folders in switchFolder
    folders_to_move = os.listdir(switchFolder)
    
    for folder_name in folders_to_move:        
        # Check if the item is a directory
        if os.path.isdir(os.path.join(switchFolder, folder_name)):
            # Get the next available number in the train_dir
            next_available_number = getNextAvailableNumber(train_dir)

            # Construct the new directory name with the next available number
            new_dir_name = f"{next_available_number}-{folder_name.split('-', 1)[1]}"

            # Move the folder from switchFolder to train_dir with the new name
            try:
                # Construct source and destination paths
                src_path = os.path.join(switchFolder, folder_name)
                dst_path = os.path.join(train_dir, new_dir_name)

                # Move the folder
                shutil.move(src_path, dst_path)

            except FileNotFoundError as e:
                print(f"Error: {e}. Source directory '{src_path}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")

            print(f"Moved '{folder_name}' to '{new_dir_name}' in '{train_dir}'")

            # Add original folder name to the list
            switch_Folders.append(folder_name.split('-', 1)[1])

    
#############################################################################################
# This function is used to move folders from 'pubfig/switchFolder' to 'pubfig/train'
#############################################################################################    
def moveBackFolders():
    global train_dir, switchFolder, switch_Folders

    # Get the list of folders in train_dir
    folders_to_move_back = os.listdir(train_dir)

    for folder_name in folders_to_move_back:
        # Check if the item is a directory
        if os.path.isdir(os.path.join(train_dir, folder_name)):
            # Check if the folder was originally from switchFolder
            if folder_name.split('-', 1)[1] in switch_Folders:  # Check if the name is in the original folders list
                # Get the next available number in the switchFolder
                next_available_number = getNextAvailableNumber(switchFolder)

                # Construct the new directory name with the next available number
                new_dir_name = f"{next_available_number}-{folder_name.split('-', 1)[1]}"

                # Move the folder back from train_dir to switchFolder with the new name
                try:
                    # Construct source and destination paths
                    src_path = os.path.join(train_dir, folder_name)
                    dst_path = os.path.join(switchFolder, new_dir_name)

                    # Move the folder
                    shutil.move(src_path, dst_path)

                except FileNotFoundError as e:
                    print(f"Error: {e}. Source directory '{src_path}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                


                print(f"Moved '{folder_name}' back to '{new_dir_name}' in '{switchFolder}'")
                
    switch_Folders.clear()
            
        
#############################################################################################
# This function is used to by the server to call the other functions in this file
# The function will detect faces in images or videos, crop the image, and move the image
# file from temp/<name> to pubfig/train
# This function can be broken into three parts:
# Part 1: Create a new directory if one is not already created for the user class
# Part 2: rename, resize, and move images with detected faces from tmp/<name> to pubfig/train/<name>
# Part 3: Remove the uploaded data from temp/<name>
#############################################################################################
def makeDir(new_folder_name):
    global new_dir, cropped, videoUploading, switchFolder, train_dir
    
    # specify the temporary directory  
    temp_dir = f"temp/{new_folder_name}"
    
    # Get the list of directories in the 'pubfig/train/' folder
    directories = next(os.walk(train_dir))[1] # [0] is the directory itself, [1] is the subdirectory, [2] are the files in the directory

    # Initialize an empty list for numbers
    numbers = []
    
    # Processed image count
    processed_images = 0
    
    # Images with detected faces
    detected_faces = 0

    # Part 1
    # Loop through all directories from the 'pubfig/train/' folder
    for dir in directories:
        # Extract the number from the directory name
        number = int(dir.split('-')[0])
        # Add the number to the list
        numbers.append(number)
        
    # Sort the numbers from 
    numbers = sorted(numbers)

    # Find the next missing number
    num = 1
    while num in numbers:
        num += 1

    # Create the new directory with the next available number if it doesn't exist
    new_dir = f"{train_dir}{num}-{new_folder_name}"

    numbers_len = len(numbers)

    # Will look through the 'pubfig/train/' file and see if the specified user name from the app is already taken
    # or if the server needs to create a new folder
    for i in range(numbers_len):
        existing_num = numbers[i]
        if os.path.exists(f"{train_dir}{existing_num}-{new_folder_name}"):
            print(f"Directory already exists: {existing_num}-{new_folder_name}")
            new_dir = f"{train_dir}{existing_num}-{new_folder_name}"
            break
        elif numbers_len-1 == i:
            print(f"Creating new directory: {new_dir}")
            os.makedirs(new_dir)
        elif i <  numbers_len-1:
            i += 1
        else:
            print("ERROR")

    # Initialize an list for the next availavle number for nameing the images files 
    numbers_face = []
    
    # Part 2
    # This will move the image from 'temp/<name>' to 'pubfig/train/name' with the proper naming format and image dimensions
    if os.path.exists(temp_dir): # Check if temp directory exists
        # go through each file in the temp directory
        for file_name in os.listdir(temp_dir):

            # construct the full path of the file
            file_path = os.path.join(temp_dir, file_name)

            # Loop through all directories
            for image in os.listdir(new_dir):
                # Extract the number from the directory name
                split_name = image.split('_')
                if len(split_name) > 1:
                    split_ext = split_name[1].split('.')
                    if len(split_ext) > 0:
                        if split_ext[0].isdigit():
                            number = int(split_ext[0])
                # Add the number to the list
                numbers_face.append(number)
            # Sort the numbers
            numbers_face.sort()

            # Find the next missing number
            num = 1
            while num in numbers_face:
                num += 1

            # Check if the file is an image
            # Note this is only for images
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                # crop the images before moving
                if videoUploading == 0:
                    processed_images += 1
                    print(f"Processing Image: {processed_images}")
                    try:
                        cropImage(file_path, new_folder_name)
                    except InvalidImage:
                        print(f"InvalidImage: Image not valid: {file_path}")
                
                # If the images has been cropped then save it to as a new file name
                if cropped == True:
                    # construct the new filename
                    base_name = os.path.basename(file_path)
                    name, ext = os.path.splitext(base_name)
                    new_name = "face_{}".format(num) + ext
                    new_path = os.path.join(new_dir, new_name)

                    # move the file to the new directory if it hasn't been removed
                    if os.path.exists(file_path):
                        try:
                            shutil.move(file_path, new_path)
                            detected_faces += 1
                        except PermissionError:
                            print("Permission Errpr")    
                        except FileNotFoundError:
                            print(f"File not found: {file_path}")
                            cropped = False

                    else:
                        print(f"File already removed: {file_path}")
        
        # Part 3
        # Retry deleting the temp directory with a delay
        # Sometimes a process could be still using one of the images and needs a second to finish
        retry_count = 3
        while retry_count > 0:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    break
                except PermissionError:
                    print(f"PermissionError: Could not delete {temp_dir}. Retrying in 1 second...")
                    time.sleep(1)
                    retry_count -= 1
        else:
            print(f"Failed to delete {temp_dir} after retrying")

    else:
        print(f"Temp directory not found: {temp_dir}")
        
    videoUploading = 0
    print(f"Detected faces: {detected_faces}")
    print("Finished makeDir function")
    return detected_faces

#############################################################################################
# # Get a count of the images for each user class in 'pubfig/train'
#############################################################################################
def countImages():
    global all_have_45, folder_name_list, all_folder_name_list
    
    # Dictionary to hold the count of images in each directory
    image_count = {}
    
    # Get a count for how many classes are elgable to train
    count_trainable = 0

    # Walk through the 'pubfig/train' directory
    for root, dirs, files in os.walk(train_dir):
        # Check if the current directory is the one to be excluded
        if root == "pubfig/train/":
            continue

        # Filter out only image files with .png or .jpg extensions
        images = [file for file in files if file.endswith(('png', 'jpg'))]
        # Update the dictionary with the count of images
        image_count[os.path.basename(root)] = len(images)
        # Check if the current directory has less than 45 images
        folder_name = root.split("/")[-1]

        # If there are more than 55 images in any user class besides Unknown, 
        # then delete random image in the class until there are only 55 images
        if len(images) > 55 and "Unknown" not in root:
            print(f"Deleting Photos for {root}")
            # Assuming images is a list of image files
            while len(images) > target_images:
                # Randomly select an image to delete
                image_to_delete = random.choice(images)

                # Construct the full path to the image
                image_path = os.path.join(root, image_to_delete)

                try:
                    # Delete the selected image
                    os.remove(image_path)

                    # Update the images list
                    images.remove(image_to_delete)

                    print(f"Deleted: {image_path}")
                except OSError as e:
                    print(f"Error deleting {image_path}: {e}")

            print(f"Number of images after deletion: {len(images)}")
            count_trainable += 1
            folder_name_list.append(folder_name)  # Add folder_name to the list
            all_folder_name_list.append(folder_name)
        
        # if a user class has less than 45 images in 'pubfig/train',
        # then notify the user that they need more images
        elif len(images) < 45 and root != "pubfig/train/":
            num_photos = len(images)
            print(f"The directory with few than 45 photos is: {root} which has {num_photos} photos")
            all_have_45 = False
            all_folder_name_list.append(folder_name)
        
        # if the user class has between 45 and 55 images,
        # then just add the user class to trainable classes for the new model 
        else:
            count_trainable += 1
            folder_name_list.append(folder_name)  # Add folder_name to the list
            all_folder_name_list.append(folder_name)

    # Print the statement only if all directories have at least 45 images
    if all_have_45:
        print("All directories have at least 45 images.")
    else:
        print("Not all directories have at least 45 images.")

    # Sort the folder_name_list in ascending order
    folder_name_list = sorted(folder_name_list)            
    all_folder_name_list = sorted(folder_name)
    
    return count_trainable, folder_name_list, all_folder_name_list


#############################################################################################
 # When ran from the python teminal  $ python main_face_rec.py it will just get the count 
 # of each user class in 'pubfig/train/' 
#############################################################################################
def main():
    countImages()
    
if __name__ == "__main__":
    main()





