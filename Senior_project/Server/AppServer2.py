#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: AppServer2
# Author of the script: Andrew Palmertree
# Date: March 9, 2024
# Version: 2.0.0
# 
# Requirments:
#     - python 3.9.18
#     - check the requirement.txt file for specific packages
#     - This file relies on the following scripts as well:
#          1. main_face_rec.py
#          2. new_model_mobileFaceNet.py
#          3. converter2.py
#          4. edge_convert2.py
#
# Usage: 
#     python appserver2
#
# Description:
#     The AppServer2 is for the BlackBeard project to be used as the server side communicate between the BlackBeard mobile app and 
#     microcontroller for the door.
#
#     App side:
#          The server will grab images or videos from the BlackBeard mobile app and process those images and videos to extract any detected 
#          faces from the data. It then sends that data to a file specified by the user to the appropriate user class in pubfig/train or 
#          pubfig/switchFolder. If the user in the app decides to train the model, the user classes in pubfig/train will get trained, but the 
#          user classes in pubfig/switchFolder won't be trained. Once the new model is trained from the new_model_mobileFaceNet.py file, the 
#          new MobileFaceNet_model_#.h5 will undergo a conversion process starting with a conversion to 
#          Converted_mobileFaceNet_model_#.tflite using the converter2.py script. The Converted_mobileFaceNet_model_#.tflite will then be 
#          converted to Converted_mobileFaceNet_model_#_edgetpu.tflite. This conversion process allows the face recognition to easily run on 
#          the microcontroller. The server will create a label file called labelmap_#.txt in the Models/labels folder, based on the user 
#          classes in pubfig/train. This label file will be used by the microcontroller for making predictions on people's faces. It will 
#          also create a new Data_File_#.txt in the Models/DataFiles folder for training a new model. The Data_File_#.txt is used to store 
#          the location and user class number. Additionally, the server will create a data_#.txt file in the Models/dataBase folder, which 
#          the microcontroller uses to determine if a given face is known or unknown.
#         
#         
#    Microcontroller Side:
#         Every ten seconds, the microcontroller will look to see if any new models have been trained. If the microcontroller finds a new 
#         model, it will download and utilize the new model immediately. The files that will be downloaded by the microcontroller include 
#         Converted_mobileFaceNet_model_#.tflite, labelmap_#.txt, and data_#.txt.
#    
#
# Notes:
# The server is running on a jupyter environment. Here are the core packages and their versions used:
# IPython          : 8.12.3
# ipykernel        : 6.29.2
# ipywidgets       : 8.1.1
# jupyter_client   : 8.6.0
# jupyter_core     : 5.7.1
# jupyter_server   : 2.12.5
# jupyterlab       : 4.0.9
# nbclient         : 0.8.0
# nbconvert        : 7.16.1
# nbformat         : 5.9.2
# notebook         : 6.5.2
# qtconsole        : 5.5.1
# traitlets        : 5.9.0
#
#############################################################################################


#============================================================================================
# Import the necessary package libraries
#============================================================================================

from flask import Flask, request, send_from_directory, jsonify, send_file
import os
import requests
import subprocess
import shutil
from PIL import Image
import shutil
import base64
import json
import cloudinary
from cloudinary.uploader import upload

#import the functions in the main_face_rec file
import main_face_rec as MFR


#============================================================================================
# Initialize variables 
#============================================================================================

# Sets the IP address for the server
# Home IP: '192.168.2.203'
# School IP: 10.101.169.96
# IP_ADDRESS = '192.168.2.203'
# IP_ADDRESS = '10.101.169.96'
# IP_ADDRESS = '10.101.167.17'
IP_ADDRESS = '192.168.182.72'


# creates an instance of the Flask class object
app = Flask(__name__)

# Get current working directory
current_directory = os.getcwd()

# Sets the directory for the images and videos to be stored before being processed
temp_folder = 'temp'
IMAGE_FOLDER = os.path.join(current_directory, temp_folder)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Directory where files will be stored for the edgetpu
# This will be the directory to store the Converted_mobileFaceNet_model_#_edgetpu.tflite files
edge_folder = 'Models\\Edge'
DOWNLOAD_FOLDER = os.path.join(current_directory, edge_folder)
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
 
# Directory where the labelmap_#.txt files will be stored 
labelmap_folder = 'Models\\labels'
DOWNLOAD_LABELS = os.path.join(current_directory, labelmap_folder)
app.config['DOWNLOAD_LABELS'] = DOWNLOAD_LABELS

# Directory where the data_#.txt files will be stored
data_folder = 'Models\\dataBase'
DOWNLOAD_DATA = os.path.join(current_directory, data_folder)
app.config['DOWNLOAD_DATA'] = DOWNLOAD_DATA

# Folder path to train user classes
train_folder = 'pubfig\\train'
TRAIN_CLASSES_FOLDER = os.path.join(current_directory, train_folder)

# Folder path to user train user classes resized
train_resized_folder = 'pubfig\\resized'
TRAIN_RESIZED_CLASSES_FOLDER = os.path.join(current_directory, train_resized_folder)

# Folder path to store the user classes that have less than 45 images to temp during the training process 
train_temp_folder = 'pubfig\\temp'
TRAIN_TEMP_FOLDER = os.path.join(current_directory, train_temp_folder)

# Folder path to user train user classes resized
train_switch_folder = 'pubfig\\switchFolder'
TRAIN_SWITCH_FOLDER = os.path.join(current_directory, train_switch_folder)

# Sets the constant ACTIVE to 1 and NOT_ACTIVE to 0
ACTIVE = 1
NOT_ACTIVE = 0

# Determines if the model should be trained
train = NOT_ACTIVE

# Used to keep track  if the user wants to train a new model from the app
status2 = NOT_ACTIVE
status3 = NOT_ACTIVE

# Status for mobile app if the model finished training 
train_status = NOT_ACTIVE

# Used to store the list of names from pubfig/train for training the new model
names = []

# Global variable to store the dictionary of directories and image counts to send to the mobile app
directory_folder_names = []
directory_image_count = []
test_dic = {}

# Used to store classes that have less than 45 images
few_images_directories = []

# Used to store classes that have at least 45 images
images_directories = []

# Used if to stop training
train_state = ' '

# Used to see if the user wants to skip the class with fewer than 45 images to train or stop the training process
skip_input = ' '

# Used to tell if you want to skip training
skip = False

# Flag to track if all directories have at least 45 images and can train
all_have_45 = True
checkCount = True

# Global variables to store the latest predictions
latest_face_predictions = []
latest_object_predictions = []


#============================================================================================
# Define used functions
#============================================================================================


#############################################################################################
# Function to get the latest version of Converted_mobileFaceNet_model_#_edgetpu.tflite
# This is used to find the next new version of the model 
#############################################################################################
def getLatestConverted():
    # Directory where the models are stored
    edge_folder = 'Models/Edge'
    
    # Initialize an empty list for model numbers
    model_numbers_converted = []
    
    # Loop through all files in the directory
    for f in os.listdir(edge_folder):
        # Check if the file is a model file
        if f.startswith('Converted_mobileFaceNet_model_'):
            # Extract the model number from the filename
            model_number = int(f.split("_")[-2].split("_edgetpu.tflite")[0])
            # Add the model number to the list
            model_numbers_converted.append(model_number)

    # Find the latest version
    if model_numbers_converted:
        latest_version_converted = max(model_numbers_converted)
    else:
        latest_version_converted = 0

    print("latest_version of edgetpu tflite: ")
    print(latest_version_converted)
    
    #Get the latest version of the converted model
    old_model_edgetpu_filename = f"Converted_mobileFaceNet_model_{latest_version_converted}_edgetpu.tflite"

    return old_model_edgetpu_filename, latest_version_converted


#############################################################################################
# This function orders the user classes in pubfig/train before the server starts training
# Orders user classes starting from 1-(last #)
#############################################################################################
def orderFolders():
    folder_path = TRAIN_CLASSES_FOLDER

    directories = []
    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isdir(full_path):
            directories.append(item)
            
    # Sort the directories
    sorted_directories = sorted(directories, key=lambda x: int(x.split('-')[0]))

    # Rename the directories sequentially
    for i, directory in enumerate(sorted_directories, start=1):
        # Get the original and new directory paths
        old_path = os.path.join(folder_path, directory)
        new_directory = f"{i}-{directory.split('-', 1)[1]}"
        new_path = os.path.join(folder_path, new_directory)

        # Rename the directory
        os.rename(old_path, new_path)
    print("Ordered folders")

    
#############################################################################################    
# This function runs after orderFolders() function
# This function does two things
# 1. It extracts detected faces from the imported images and videos from the mobile app 
#    and saves the the date in pubfig/train or pubfig/switchFolder
#    If the user classes folder is already created and set to not to be trained then the user classes will located in pubfig/train
# 2  If a user class in pubfig/train contains more than 55 images, then this function will delete random photos 
#    until there are only 55 photos in the folder
#############################################################################################
def deleteImages():
    # Command to run your Python script
    command = ["python", "main_face_rec.py"]

    # Run the command as a subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for the subprocess to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
    
    
#############################################################################################
# This function is used to resize images in the pubfig/train to the appropriate size for the face recognition model
# The resized images will be saved in pubfig/resized
#############################################################################################
def resizeImages():
    print("Resizing images")
    # Define the folder paths and desired image size
    input_folder = TRAIN_CLASSES_FOLDER
    output_folder = TRAIN_RESIZED_CLASSES_FOLDER
    desired_size = (96, 112)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through all files and directories in the input directory
    for root, dirs, files in os.walk(input_folder):
        # Process each file
        for filename in files:
            # Check if the file is an image
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Construct the input and output paths
                input_path = os.path.join(root, filename)
                # Construct the output path by replacing input folder path with output folder path
                output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))

                # Create directory structure in output folder if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Open the image
                with Image.open(input_path) as img:
                    # Resize the image
                    resized_img = img.resize(desired_size)
                    # Save the resized image
                    resized_img.save(output_path)

                    
#############################################################################################
# This function checks each user class folder in pubfig/train to see if all user classes have at least 45 images
# If any user class has fewer than 45 images, the server will notify the user on the mobile app which users do 
# not have enough images to train the new model.    
#############################################################################################
def checkImageCount():
    global checkCount, few_images_directories, images_directories, directory_image_count, directory_folder_names
    # Get a list of all directories in the folder
    directories = []
    
    directory_image_count = []
    directory_folder_names = []
    test_dic = {}
    
    folder_path = TRAIN_CLASSES_FOLDER
    
    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isdir(full_path):
            directories.append(item)

    # Sort the directories
    sorted_directories = sorted(directories, key=lambda x: int(x.split('-')[0]))

    # Create a list to store directories with less than 45 images
    few_images_directories = []
    
    # Create a list to store directories with at least 45 images
    images_directories = []

    # Iterate through sorted directories
    for directory in sorted_directories:
        # Get the full path of the directory
        full_directory_path = os.path.join(folder_path, directory)

        # Count the number of files in the directory
        file_count = len(os.listdir(full_directory_path))

        # Print the directory name and file count
        print(f"Folder '{directory}' has {file_count} files.")
        
        # Add directory and its image count to the dictionary
        directorySplit = directory.split('-')[1]
        if directorySplit == "Unknown":
            continue
        else:
            directory_folder_names.append(directorySplit)
            directory_image_count.append(file_count)

        # Check if the directory has less than 45 images
        if file_count < 45:
            few_images_directories.append(directory)
            checkCount = False
        else:
            images_directories.append(directory)

    # Print directories with less than 45 images
    if few_images_directories:
        print("\nDirectories with less than 45 images:")
        for directory in few_images_directories:
            print(directory)
    else:
        print("\nAll directories have 45 or more images.")
        
        
#############################################################################################
# Function to train the new model
# This grabs the user input from the mobile app
# If the user confirms to train the new model, then the new_model_mobileFaceNet.py will run
# The output from this function will create a file called MobileFaceNet_model_#.h5
#############################################################################################
def train_model():
    global trained, train_state, skip_input, skip
    
    source_folder = TRAIN_CLASSES_FOLDER
    destination_folder = TRAIN_TEMP_FOLDER
    
    if checkCount == True:
        # Gets the input from the user on the mobile app to confirm the training the new model
        print("Using first input from app")
        if status2 == '1':
            train_state = 'y'
        else:
            train_state = 'n'
            print("WARNING SOMETHING FAILED")
    
    # if the user confirms to train the new model, the python script new_model_mobileFaceNet.py will train the new model
    if train_state == 'y':
        print("Training")
        # Command to run python script to train a new face recognition model
        command = ["python", "new_model_mobileFaceNet.py"]

        # Run the command as a subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the subprocess to finish and get the output
        stdout, stderr = process.communicate()

        # Print the output
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())
        trained = True
        
    elif train_state == 'n':
        skip = True
    else:
        print("Not able to train.")
        print("Do you want skip training the following user(s):")
        print(few_images_directories)
        print("second user input from app")
        if status3 == '1':
            skip_input = 'y'
        else:
            skip_input = 'n'
            print("WARNING SOMETHING WENT WRONG")
            
        # Trains the new model if the user want to skip the user classes in pubfig/train that have less than 45 images            
        if skip_input == 'y':
            print("preparing")
            # Move directories with fewer than 45 images to the destination folder
            for directory in few_images_directories:
                source_dir = os.path.join(source_folder, directory)
                destination_dir = os.path.join(destination_folder, directory)
                shutil.move(source_dir, destination_dir)
                print(f"Moved '{directory}' to '{destination_folder}'")
                
                for folder in few_images_directories:
                    folder = f'{folder}'
                    path = os.path.join(TRAIN_RESIZED_CLASSES_FOLDER, folder)
                    # Check if the directory exists before attempting to delete
                    if os.path.exists(path):
                        shutil.rmtree(path)
                        print(f"Deleted '{path}'")
                    else:
                        continue

            print("\nDirectories moved successfully.")
            
            print("training")
            # Command to run the python script named new_model_mobileFaceNet.py
            command = ["python", "new_model_mobileFaceNet.py"]

            # Run the command as a subprocess
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for the subprocess to finish and get the output
            stdout, stderr = process.communicate()

            # Print the output
            print("stdout:", stdout.decode())
            print("stderr:", stderr.decode())
            trained = True
            
        else:
            print("skipping training")
            skip = True

            
#############################################################################################
# Function to convert the trained model from MobileFaceNet_model_#.h5 to 
# Converted_mobileFaceNet_model_#.tflite using the python script named converter2.py
#############################################################################################
def convert():
    global converted, trained
    if trained == True:
        print("Converting")

        # Command to run the python script named converter2.py
        command = ["python", "converter2.py"]

        # Run the command as a subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the subprocess to finish and get the output
        stdout, stderr = process.communicate()

        # Print the output
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())

        converted = True 
    else:
        print("Not able to convert.")

#############################################################################################
# This function converts Converted_mobileFaceNet_model_#.tflite to 
# Converted_mobileFaceNet_model_#_edgetpu.tflite using the python script called 
# edge_convert2.py
# Note: This is to quantize the model and is the last conversion step in the model 
#############################################################################################        
def edge_convert():
    global quantized
    if converted == True:
        print("Quantizing")
        # Command to run the python script named converter2.py
        command = ["python", "edge_convert2.py"]

        # Run the command as a subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the subprocess to finish and get the output
        stdout, stderr = process.communicate()

        # Print the output
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())
        quantized = True
    else:
        print("Was not able to quantize")

        
#############################################################################################        
# This function is used to create a data base file called data_#.txt to store 
# the predicted weights for each class from pubfig/train 
#############################################################################################        
def dataBase():
    global dataBased
    if quantized == True:
        print("Creating data base file")
        # Define the command to run your Python script
        command = ["python", "dataBase.py"]

        # Run the command as a subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the subprocess to finish and get the output
        stdout, stderr = process.communicate()

        # Print the output
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())
        dataBased = True
    else:
        print("Was not able to create data base")
        
        
#############################################################################################
# Removes the resized image folder after training
#############################################################################################   
def removeResizedFolders():
    # Delete old folders from pubfig\\resized\\   
    folder_path = TRAIN_RESIZED_CLASSES_FOLDER

    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)

            
#############################################################################################   
# Get a list of directory names in the temp/ folder to later process to pubfig/train
#############################################################################################   
def listNames():

    # specify the path to your temp folder
    path = IMAGE_FOLDER
    
    # use os.listdir to get all items in the directory
    items = os.listdir(path)
    
    # create an empty list to store the folder names
    folder_names = []
    
    # go through each item in the directory
    for item in items:
        # construct the full path of the item
        full_path = os.path.join(path, item)
    
        # check if the item is a directory
        if os.path.isdir(full_path):
            # if it is, add its name to our list
            folder_names.append(item)

    return folder_names


############################################################################################# 
# Goes through the specified directory and find the first image
# This will take each folder in pubfig/train and find the first image
# This is used for the app to show the user's faces when selecting and deselecting classes 
############################################################################################# 
def load_first_image_from_directories(root_path):
    # A list to store the first image from each directory
    first_images = []  
    
    # Go through each directory and its contents
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Print directory names for debugging
        print("Directory path:", dirpath)
        print("Directory names:", dirnames)
        
        # Exclude directories with names containing 'Unknown'
        for dirname in dirnames:
            if dirname.endswith('-Unknown'):
                dirnames.remove(dirname)
        
        # Go through each file in the directory
        for filename in filenames:
            # Check if the file is an image (ends with .jpg or .png)
            if filename.lower().endswith(('.jpg', '.png')):
                # Construct the full path of the image
                image_path = os.path.join(dirpath, filename)
                # Add the image path to the list
                first_images.append(image_path)
                # Stop searching for more images in this directory
                break  
    
    # Return the list of paths to the first images in each directory
    return first_images


############################################################################################# 
# Configuration settings for Cloudinary
############################################################################################# 
cloudinary.config(
  cloud_name = 'andrewscloud',
  api_key = '894696721332575',
  api_secret = 'xxtN9sRYiwhOKo_10Lyuk8tiExA'
)


############################################################################################# 
# This function will upload an image to the cloudinary server to obtain a URL to share
# with the mobel app
# For the server, it will only process the first image from each folder in pubfig/train
############################################################################################# 
def upload_image_to_cloudinary(image_path):
    """Uploads an image to Cloudinary and returns the URL."""
    try:
        # Upload the image to Cloudinary
        result = upload(image_path)
        
        # Extract the URL from the upload result
        return result['secure_url']
    except Exception as e:
        print("Error uploading image to Cloudinary:", e)
        return None    
    

#============================================================================================
# Define used functions for 'GET' and 'POST' in Flask
#============================================================================================ 

############################################################################################# 
# This FLask function is used to obtain a video file from the mobile app and 
# the user name that correlates with the video
#
# Once the function confirms that the mobile app sent a video and user name, then the
# function will process the video to extract faces from it and save the data to a folder 
# named after the user class in pubfig/train or pubfig/switchFolder
# Note: pubfig/switchFolder is only for user classes that are selected not to be trained
############################################################################################# 
#video function   
@app.route('/video', methods=['POST'])
def video():
    # This moves the folder from pubfig/switchFolder to pubfig/train
    MFR.moveFolders()
    
    #Reset the names list
    names = []

    print("Received data:", request.data)
    print("Received args:", request.args)

    photo_url = request.data.decode('utf-8')

    #Receives the query parameter in the POST and sets that to be the users name
    file_name = request.args.get('FileName')

    #If both are received the process will begin
    if photo_url and file_name:
        #Moves to folder in the path set above named after the users set File
        folder_path = os.path.join(app.config['IMAGE_FOLDER'], file_name)
        #If this folder does not exist this will creates it 
        if not os.path.exists(folder_path):
          os.makedirs(folder_path)

        #Fetchs the image based on the uploaded URL
        response = requests.get(photo_url)

        #If this process is successful:
        if response.status_code == 200:
            #We now save the image as "FileName(n).png" in the folder
            #where file name is the user and n is the number of the photo uploaded
            image_number = len(os.listdir(folder_path))
            image_filename = f"{file_name}({image_number}).MP4"
            image_path = os.path.join(folder_path, image_filename)


            with open(image_path, 'wb') as f:
              f.write(response.content)

            # This extract the faces from the video
            MFR.videoFaceExtraction(image_path, file_name)

            names.append(listNames())
            # go through each name in the names list
            for name_list in names:
                # go through each name in the name_list
                for name in name_list:
                    # Makes the folder for the specifc user class or just add to previous folder if one is already created
                    detected_faces = MFR.makeDir(name)

            # move specific folders from pubfig/train back to pubfig/switchFolder
            MFR.moveBackFolders()
            return f"Success! Detected {detected_faces} faces!", 200
        else:
            MFR.moveBackFolders()
            return "Failed to fetch the Video", 400
    else:
        MFR.moveBackFolders()
        return "Invalid request data", 400

    
############################################################################################# 
# Function to move folders from 'pubfig/train' to 'pubfig/switchFolder'
#############################################################################################     
@app.route('/move_folders_app', methods=['POST'])
def move_folders_app():    
    
    # Receives the query parameter in the POST and sets to move the folders
    move_status = request.args.get('Move_Status')
    
    if move_status == '1':
        # This moves the folder from pubfig/switchFolder to pubfig/train 
        MFR.moveFolders()
        return "Images saved successfully", 200
    else:
        return "Failed to move the images", 400
    
    
############################################################################################# 
# This FLask function is used to obtain file images from the mobile app and 
# the user name that correlates with the images
#
# Once the function confirms that the mobile app sent the images and user name, then the
# function will store the data in the temp/<User_Name> folder
############################################################################################# 
@app.route('/upload', methods=['POST'])
def upload_image():
    global image_move_status
    
    # Receives a URL from the mobile app in the POST body and converts it to proper characters--
    photo_url = request.data.decode('utf-8')

    # Receives the query parameter in the POST and sets that to be the users name
    file_name = request.args.get('FileName')

    # If both are received the process will begin
    if photo_url and file_name:
        # Moves to the imported images to temp/<User_Name>
        folder_path = os.path.join(app.config['IMAGE_FOLDER'], file_name)
        # If this folder does not exist this will creates it 
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except FileExistsError:
            # Directory already exists, do nothing
            pass
    
            
        #Fetchs the image based on the uploaded URL
        response = requests.get(photo_url)

        #If this process is succesful:
        if response.status_code == 200:
            # We now save the image as "FileName(n).png" in the folder
            # where file name is the user and n is the number of the photo uploaded
            image_number = len(os.listdir(folder_path))
            image_filename = f"{file_name}({image_number}).png"
            image_path = os.path.join(folder_path, image_filename)

            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            return "Images saved successfully", 200
        else:
            return "Failed to fetch the images", 400
    else:
        return "Invalid request data", 400

    
############################################################################################# 
# This FLask function is used to process the images from temp/<User_Name> folder to 
# extract faces from it and save the data to a folder named after the user class in 
# pubfig/train or pubfig/switchFolder
############################################################################################# 
@app.route('/crop_images', methods=['POST'])
def crop_images():
    global image_move_status
    
    names = []
    
    #Receives the query parameter in the POST and sets that to be the users name
    file_name = request.args.get('FileName')

    #If both are received the process will begin
    if file_name:
        names.append(listNames())
        name = names[0][0]
        detected_faces = MFR.makeDir(name)

        # move specific folders from pubfig/train back to pubfig/switchFolder
        MFR.moveBackFolders()
        return f"Success! Detected {detected_faces} faces!", 200
    else:
        MFR.moveBackFolders()
        Print("test")
        return "Failed to crop the images", 400

    
############################################################################################# 
# This function is used when the user request to train a new model on the app
############################################################################################# 
## train function
@app.route('/train', methods=['POST'])
def train():
    global train_state, skip_input, few_images_directories, skip, checkCount, train_status
    
    # Photo path    
    train = request.args.get('Train')
    
    # If the user request to train a new model on the app this will go through all the code
    if train=='1':
        print("In Train Function")
        # Odrer all the folders in the pubfig/train
        orderFolders()
        # delete randome images from user classes that have more than 55 images
        deleteImages()
        # Resize images in the input folder (pubfig/train) and save to the output folder (pubfig/resized)
        resizeImages()
        # Checks and sees if all user classes have at least 45 images each
        checkImageCount()
        # This function trains the new model
        train_model()
        
        # This converts the .h5 model into edgetpu.tflite model if the user is okay training the new model
        if skip != True:
            convert()
            edge_convert()
            dataBase()

            # This will move the files from pubfig/temp back to pubfig/train
            # pubfig/temp is where the user classes with less than 45 images are stored when the user
            # wants to train a new model and skip the classes with fewer than 45 images
            if skip_input == 'y':
                print("moving files back")
                source_folder = TRAIN_TEMP_FOLDER
                destination_folder = TRAIN_CLASSES_FOLDER

                # Move directories with fewer than 45 images to the destination folder
                for directory in few_images_directories:
                    source_dir = os.path.join(source_folder, directory)
                    destination_dir = os.path.join(destination_folder, directory)
                    shutil.move(source_dir, destination_dir)
                    print(f"Moved '{directory}' to '{destination_folder}'")

                print("\nDirectories moved successfully.")
        
        else:
            print("Skipping Training")
        
        # reset all the variables and folders
        train_status=1
        removeResizedFolders()
        checkCount = True
        train_state = ' '
        skip_input = ' '
        few_images_directories = []
        skip = False    
        train = 0
        return "Successfully Trained ", 200
    else:
        return "Invalid request data", 400
    

############################################################################################# 
# Used for the microcontroller to download the latest edgetpu file from the server
############################################################################################# 
@app.route('/download', methods=['POST'])
def upload_file():
    
    # Check if 'file' is not in the request files
    if 'file' not in request.files:
        return 'No file part'
    
    # Retrieve the file from the request
    file = request.files['file']
    
    # Check if the filename is empty
    if file.filename == '':
        return 'No selected file'
    
    # Check if a file object is present
    if file:
        file.save(os.path.join(app.config['DOWNLOAD_FOLDER'], file.filename))
        return 'File uploaded successfully'
    
@app.route('/downloads/<filename>')
def uploaded_file(filename):
    # Use send_from_directory to send the edgetpu file from the DOWNLOAD_FOLDER to the client
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)


############################################################################################# 
# Used for the microcontroller to download the latest label file from the server
############################################################################################# 
@app.route('/download_label', methods=['POST'])
def upload_file_label():
    # Check if 'file' is not in the request files
    if 'file' not in request.files:
        return 'No file part'
    
    # Retrieve the file from the request
    file = request.files['file']
    
    # Check if the filename is empty
    if file.filename == '':
        return 'No selected file'
    
    # Check if a file object is present
    if file:
        file.save(os.path.join(app.config['DOWNLOAD_LABELS'], file.filename))
        return 'File uploaded successfully'

@app.route('/download_label/<filename>')
def uploaded_file_label(filename):
    # Use send_from_directory to send the label file from the DOWNLOAD_LABELS to the client
    return send_from_directory(app.config['DOWNLOAD_LABELS'], filename)


############################################################################################# 
# Used for the microcontroller to download the latest data file from the server
############################################################################################# 
@app.route('/download_data', methods=['POST'])
def upload_file_data():
    
    # Check if 'file' is not in the request files
    if 'file' not in request.files:
        return 'No file part'
    
    # Retrieve the file from the request
    file = request.files['file']
    
    # Check if the filename is empty
    if file.filename == '':
        return 'No selected file'
    
    # Check if a file object is present
    if file:
        file.save(os.path.join(app.config['DOWNLOAD_DATA'], file.filename))
        return 'File uploaded successfully'

@app.route('/download_data/<filename>')
def uploaded_file_data(filename):
    # Use send_from_directory to send the data file from the DOWNLOAD_DATA to the client
    return send_from_directory(app.config['DOWNLOAD_DATA'], filename)


############################################################################################# 
# Used for the microcontroller to get the latest version of the face recognition model from 
# the server
# This will determine if the microcontroller needs to request the new model's files
############################################################################################# 
@app.route('/latest_version', methods=['GET'])
def latest_version():
    old_model_edgetpu_filename, latest_version_converted = getLatestConverted()
    return old_model_edgetpu_filename


############################################################################################# 
# Used by the mobile app to determine if the training process is completed
############################################################################################# 
@app.route('/train_stats', methods=['GET'])
def train_stats():
    train_status_update = train_status
    
    if train_status_update == 1:
        return "1"
    else:
        return "Training"
    

############################################################################################# 
# Used by the mobile app to determine the user claseses in pubfig/train and the images count
# for each user class
#############################################################################################  
@app.route('/get_countIMages_App', methods=['GET'])
def get_countIMages_App():
    global train_status
    
    # Call the function to check image counts
    checkImageCount()
    
    # reset train status
    train_status = 0
    
    # Convert the directory image count and folder names to JSON format
    json_data = json.dumps({"directory_image_count": directory_image_count, "directory_folder_names" : directory_folder_names})
    
    return json_data


############################################################################################# 
# This will tell the app the user classes in pubfig/train that have less than 45 images
# and if the user wants to skip those classes and continue training or cancel the training
# process
############################################################################################# 
@app.route('/get_less_45', methods=['GET'])
def get_countIMages_App2():
    
    # Call the function to check image counts
    checkImageCount()
    
    directory_names = []

    # Iterate through the list of directory names
    for directory in few_images_directories:
        directory_split = directory.split('-')[1].strip()
        
        # Check if the directory name is 'Unknown'
        if directory_split == "Unknown":
            continue
        else:
            directory_names.append(directory_split)
            
    print(directory_names)

    # Now, you can append the directory names without the count to the new list
    son_data = json.dumps({"directory_names": directory_names})

    # Finally, you can return the JSON data
    return son_data


############################################################################################# 
# Used by the mobile app to determine if the sever got the training request
############################################################################################# 
@app.route('/train_stats2', methods=['POST'])
def train_stats2():
    global status2
    
    # Retrieve the status2 parameter from the request's arguments
    status2 = request.args.get('status2')

    if status2=='1':
        print("status2 successful")
        return "Input Received"
    
    else:
        print("error")
        return "Not Successfully"
    
    
############################################################################################# 
# Used by the mobile app to determine if the sever started training
############################################################################################# 
@app.route('/train_stats3', methods=['POST'])
def train_stats3():
    global status3
    
    # Retrieve the status3 parameter from the request's arguments   
    status3 = request.args.get('status3')

    if status3 =='1':
        print("status3 successful")
        return "Training"
    
    else:
        print("error")
        return "Not Successfully"    


############################################################################################# 
# Used by the mobile app to get the first image from each user class in pubfig/train 
############################################################################################# 
@app.route('/image', methods=['GET'])
def get_first_image():
    root_directory = TRAIN_CLASSES_FOLDER
    
    # Get the paths of the first image from each subdirectory
    first_image_paths = load_first_image_from_directories(root_directory)
    
    image_url_list = []
    
    # Upload each image to Cloudinary and store its URL
    for image_path in first_image_paths:
        image_url = upload_image_to_cloudinary(image_path)
        if image_url:
            image_url_list.append(image_url)
            
    # Convert the list of image URLs to JSON format
    json_data = json.dumps({'images': image_url_list})
    
    return json_data


############################################################################################# 
# Used by the mobile app to move user classes' folders from pubfig/train to 
# pubfig/switchFolder
# This is used when the user selects and deselects classes to train on the app
############################################################################################# 
@app.route('/user_select', methods=['POST'])
def user_select():
    
    # Function to find the next available number for folder naming
    def find_next_available_number(directory, moving_to_train):
        existing_numbers = []
        for folder in os.listdir(directory):
            if '-' in folder:
                try:
                    number = int(folder.split('-')[0])
                    existing_numbers.append(number)
                except ValueError:
                    continue  # Skip folders that don't start with a number

        if moving_to_train:
            existing_numbers.sort()
            next_number = 1
            for number in existing_numbers:
                if number != next_number:
                    return next_number
                next_number += 1
            return next_number
        else:
            return max(existing_numbers) + 1 if existing_numbers else 1

    # Function to reassign folder numbers to maintain sequential order   
    def reassign_folder_numbers(directory):
        folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
        folders.sort(key=lambda x: int(x.split('-')[0]))

        for i, folder in enumerate(folders):
            current_number = int(folder.split('-')[0])
            if current_number != i + 1:
                new_name = f"{i + 1}-{folder.split('-', 1)[1]}"
                os.rename(os.path.join(directory, folder), os.path.join(directory, new_name))

    # Retrieve the selected parameter from the request's arguments            
    selected = request.args.get('selected')

    if selected == '':
        print("Error: No selection made.")
        return "Not Successfully"
    else:
        print("Selection successful:", selected)

        formatted_json_string = f"[{selected}]"

        try:
            selected_data = json.loads(formatted_json_string)

            source_dir = TRAIN_CLASSES_FOLDER
            destination_dir = TRAIN_SWITCH_FOLDER

            # Function to move folders based on their active status
            # If status equals True, then move the folder from pubfig/train to pubfig/switchFolder
            # If status equals False, then move the folder from pubfig/switchFolder to pubfig/train
            def move_folders(based_on_active):
                if based_on_active:
                    current_source_dir = destination_dir
                    current_destination_dir = source_dir
                else:
                    current_source_dir = source_dir
                    current_destination_dir = destination_dir

                moving_to_train = current_destination_dir.endswith("train")

                for entry in selected_data:
                    name = entry.get("name")
                    active = entry.get("active")

                    if active == based_on_active:
                        for folder in os.listdir(current_source_dir):
                            if folder.endswith(f"-{name}"):
                                source_path = os.path.join(current_source_dir, folder)
                                next_number = find_next_available_number(current_destination_dir, moving_to_train)
                                new_folder_name = f"{next_number}-{name}"
                                destination_path = os.path.join(current_destination_dir, new_folder_name)
                                print(f"Moving {source_path} to {destination_path}")
                                shutil.move(source_path, destination_path)
                                break  # Assuming only one folder per name

            move_folders(False)  # Move inactive folders first
            move_folders(True)   # Then move active folders

            # Reassign folder numbers only when moving to the switchFolder to ensure sequential order
            if destination_dir.endswith("switchFolder"):
                reassign_folder_numbers(destination_dir)

            print("Folder move based on active status completed successfully.")
            return "Successful"

        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return "Not Successfully"

        
############################################################################################# 
# Used by the mobile app to see which user classes are in pubfig/train and 
# pubfig/switchFolder
# If the user class is in the pubfig/train, then the switch on the app will be active
# If the user class is in the pubfig/switchFolder, the switch on the app will be inactive 
#############################################################################################         
@app.route('/status_switchs', methods=['GET'])
def status_switchs():
    
    # Define source and destination directories
    source_dir = TRAIN_CLASSES_FOLDER
    destination_dir = TRAIN_SWITCH_FOLDER

    # Function to retrieve user folders and their details
    def get_user_folders(directory_path, active):
        names = []
        actives = []
        image_counts = []
        first_image_paths = []
        
        # Check if the directory exists
        if os.path.exists(directory_path):
            for folder_name in os.listdir(directory_path):
                if '-Unknown' in folder_name:
                    continue
                if '-' in folder_name:
                    _, name = folder_name.split('-', 1)
                    names.append(name)
                    actives.append(active)
                    folder_path = os.path.join(directory_path, folder_name)
                    image_count = 0
                    
                    # Iterate through files in the folder
                    for entry in sorted(os.listdir(folder_path)):
                        full_path = os.path.join(folder_path, entry)
                        if os.path.isfile(full_path):
                            image_count += 1
                            if image_count == 1:  # First image
                                first_image_paths.append(full_path)
                    image_counts.append(image_count)
        return names, actives, image_counts, first_image_paths

    # Function to upload an image to Cloudinary and retrieve its URL
    def upload_image_to_cloudinary(image_path):
        try:
            result = upload(image_path)
            return result['secure_url']
        except Exception as e:
            print(f"Error uploading image to Cloudinary: {e}")
            return None

    # Retrieve user folders and their details for switch and train directories    
    switch_names, switch_actives, switch_image_counts, switch_first_images = get_user_folders(destination_dir, False)
    train_names, train_actives, train_image_counts, train_first_images = get_user_folders(source_dir, True)

    # Combine details of all users from both directories
    all_names = switch_names + train_names
    all_actives = switch_actives + train_actives
    all_image_counts = switch_image_counts + train_image_counts
    all_first_images = switch_first_images + train_first_images

    # Upload first images to Cloudinary and store their URLs
    image_url_list = [upload_image_to_cloudinary(path) for path in all_first_images if path]

    # Create JSON object containing details of all users
    all_users_json = jsonify({
        "name": all_names,
        "active": all_actives,
        "directory_image_count": all_image_counts,
        'images': image_url_list
    })

    return all_users_json


############################################################################################# 
# Used by the microcontroller to send frames of the video to the server
############################################################################################# 
@app.route('/video_upload', methods=['POST'])
def video_upload():
    global latest_face_predictions, latest_object_predictions
    
    # Extract video frame from the request
    video_frame = request.files.get('video_frame', None)
    
    # Attempt to extract and decode the face and object predictions
    face_predictions_str = request.form.get('face_predictions', "[]")  # Default to empty list as string
    object_predictions_str = request.form.get('object_predictions', "[]")  # Default to empty list as string
    
    # Convert the JSON strings back to Python objects (lists/dicts)
    try:
        face_predictions = json.loads(face_predictions_str)
        object_predictions = json.loads(object_predictions_str)
    except json.JSONDecodeError:
        print("Error decoding JSON from predictions")
        return "Error decoding predictions", 400

    # For demonstration, just print received predictions
    print("Received face predictions:", face_predictions)
    print("Received object predictions:", object_predictions)
    
    # Update the global variables with the latest predictions
    latest_face_predictions = face_predictions
    latest_object_predictions = object_predictions
    
    # Save video frame for processing or demonstration
    if video_frame:
        frame_save_path = "received_frame.jpg"
        video_frame.save(frame_save_path)
        print(f"Video frame saved to {frame_save_path}")
        return jsonify({"status": "Success"})
    else:
        print("No video frame received")
        return jsonify({"status": "Not Successfully"}), 400

    
############################################################################################# 
# Used by the mobile app to get the frames of the video from microcontroller
############################################################################################# 
@app.route('/get_video', methods=['GET'])
def get_video_frame():
    frame = "received_frame.jpg"
    return send_file(frame, mimetype='image/jpeg')


############################################################################################# 
# Used by the mobile app to get class names and predictions of objects and faces
############################################################################################# 
@app.route('/get_predictions_video', methods=['GET'])
def get_predictions_video():
    response_data = jsonify({
        "face_predictions": latest_face_predictions,
        "object_predictions": latest_object_predictions
    })
    return response_data


############################################################################################# 
# Functions used by the app to tell the server if the microcontroller needs to have live
# feedback
############################################################################################# 
@app.route('/set_video_active', methods=['POST'])
def set_video_active():
    global video_status
    video_status = request.args.get('video_status')
    print(video_status)
    if video_status is not None:
        return "Success", 200
    else:
        return "Error", 400
    
    
############################################################################################# 
# Used to determine if the microcontroller needs to upload frames to the server
############################################################################################# 
@app.route('/get_video_active', methods=['GET'])
def get_video_active():
    global video_status
    try:
        if video_status is not None:
            return video_status
        else:
            return "Error", 400
    except NameError:
        # The case where video_status is not defined
        return "Waiting to activate live video feed from user", 200

    
############################################################################################# 
# Function used if no images are being uploaded from microcontroller to app
#############################################################################################     
@app.route('/favicon.ico')
def favicon():
    return '', 204 


############################################################################################# 
# Sets the server's IP address for the app and microcontroller to connect to 
############################################################################################# 
if __name__ == '__main__':
    
    # Sets the IP address
    app.run(host=IP_ADDRESS, port=5353)

    
