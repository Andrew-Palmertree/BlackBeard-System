#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: dataBase.py
#
#
# Date: March 9, 2024
# Version: 1.0.0
# 
# Requirments:
#     - python 3.9.18
#     - Check the requirement.txt file for specific packages. 
#
# Usage: 
#     Used for the AppServer2 script to create a file called data base which contains the 
#
# Description:
#     The dataBase script is for the BlackBeard project to be used on the server side.
#     This file will take the new model Converted_mobileFaceNet_model_{#}.tflite and create 
#     a file with 128 features that will be used to compare with the live feed to determine
#     if a face is a known face.
#
# Notes:
#  
#
#############################################################################################


#============================================================================================
# Import the necessary package libraries
#============================================================================================
import os
import random
import numpy as np
import tensorflow as tf
import cv2


#============================================================================================
# Initialize variables 
#============================================================================================


# Get current working directory
current_directory = os.getcwd()

# Sets the Converted .tflite model directory path
convert_folder = 'Models/Converted/'
CONVERT_FOLDER = os.path.join(current_directory, convert_folder)

# Sets the data folder directory path
data_folder = 'Models/DataFiles/'
DATA_FOLDER = os.path.join(current_directory, data_folder)

# Sets the data base folder directory path
dataBase_folder = 'Models/dataBase/'
DATABASE_FOLDER = os.path.join(current_directory, dataBase_folder)


#============================================================================================
# Define used functions
#============================================================================================


#############################################################################################
# Function to get the latest version of the trained models
#############################################################################################
def latestModelVersion():
    # used to check if there is a previous trained model to name the new model with the correct version
#     no_previous_models = 0
    
## Gets the latest version of the .h5 MobileFaceNEt model

    # Directory where the models are stored
    models_directory_models = 'Models/'
    
    # Initialize an empty list for model numbers
    model_numbers_models = []
    
    # Loop through all files in the directory
    for f in os.listdir(models_directory_models):
        # Check if the file is a model file
        if f.startswith('MobileFaceNet_model_'):
            # Extract the model number from the filename
            model_number = int(f.split("_")[-1].split(".h5")[0])
            # Add the model number to the list
            model_numbers_models.append(model_number)

    # Find the latest version
    if len(model_numbers_models) > 0:
        latest_version = max(model_numbers_models)
    else:
        latest_version = 0
        
    return latest_version


#############################################################################################
# Function to preprocess the images before the model creates the 128 features of the image
#############################################################################################
def preprocess(img):
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


#############################################################################################
# Function reates the 128 features of a image for each class and savev it to a file
#############################################################################################
def main():
    
    # Get the latest version of the model
    latest_version = latestModelVersion()
    model_path = os.path.join(CONVERT_FOLDER, f'Converted_mobileFaceNet_model_{latest_version}.tflite') 
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # path to the text file
    data_file_path = os.path.join(DATA_FOLDER, f'Data_File_{latest_version}.txt')
    print(latest_version)

    # Create a dictionary to store file paths by class
    files_by_class = {}

    # Read the contents of the text file and parse lines
    with open(data_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_path, class_id = line.strip().split()
            if class_id not in files_by_class:
                files_by_class[class_id] = []
            files_by_class[class_id].append(file_path)

    # Select one random file from each class
    random_files = {}
    for class_id, file_paths in files_by_class.items():
        random_files[class_id] = random.choice(file_paths)
        
    image_output_data = []    
    # Print the selected random files
    for class_id, file_path in random_files.items():
        print(f"Class {class_id}: {file_path}")
        # process one image from each class
        img = preprocess(cv2.imread(file_path))
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        image_output = interpreter.get_tensor(output_details[0]['index'])
        image_output = image_output / np.expand_dims(np.sqrt(np.sum(np.power(image_output, 2), 1)), 1)
        image_output_data.append(image_output)

    # Save embedding tests to file
    file = os.path.join(DATABASE_FOLDER, f'data_{latest_version}.txt')

    for j, image_output in enumerate(image_output_data):
        if j == 0:
            np.savetxt(file, image_output)
        else:
            with open(file, 'ab') as f:
                np.savetxt(f, image_output)
    
    # clear the variables    
    image_output_data = []
    files_by_class = {}
    random_files = {}


if __name__ == '__main__':
    main()