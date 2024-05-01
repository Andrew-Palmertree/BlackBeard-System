#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: converter2.py
# Author of the script: Andrew Palmertree
# Date: March 9, 2024
# Version: 2.0.0
# 
# Requirments:
#     - python 3.9.18
#     - check the requirement.txt file for specific packages
#
# Usage: 
#     used for the AppServer2 script
#
# Description:
#     The converter2 script is for the BlackBeard project to be used on the server side to
#     convert the .h5 model create by the new_model_mobileFaceNet.py to a .tflite file.
#
#
#
# Notes:
# keras uses the TFLite Converter to convert the .h5 models to tflite model
#
#############################################################################################


#============================================================================================
# Import the necessary package libraries
#============================================================================================
import tensorflow as tf
import keras
import keras_vggface
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.utils.data_utils import get_file
import keras_vggface.utils
import os
import os.path


#############################################################################################
# Function to convert .h5 models to .tflite models using tensorflow and keras
#############################################################################################
def main():

    # Directory where the models are stored
    models_directory_converted = 'Models/Converted'
    
    # Initialize an empty list for model numbers
    model_numbers_converted = []
    
    # Loop through all files in the directory
    for f in os.listdir(models_directory_converted):
        # Check if the file is a model file
        if f.startswith('Converted_mobileFaceNet_model_'):
            # Extract the model number from the filename
            model_number = int(f.split("_")[-1].split(".tflite")[0])
            # Add the model number to the list
            model_numbers_converted.append(model_number)

    # Find the latest version
    if len(model_numbers_converted) > 0:
        latest_version_converted = max(model_numbers_converted)
        # Increment the model version
        new_version_converted = latest_version_converted + 1
    else:
        latest_version_converted = 0
        new_version_converted = latest_version_converted

    print(f"latest_version of normal tflite: {latest_version_converted}")
    
    # Create the new model file name
    new_model_filename = os.path.join(models_directory_converted, f"Converted_mobileFaceNet_model_{new_version_converted}.tflite")

    print(f"new_version of normal tflite: {new_version_converted}")

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
        latest_version_models = max(model_numbers_models)
    else:
        latest_version_models = 0

    print("latest_version of .h5: ")
    print(latest_version_models)

    # Set the environment variable to allow duplicated libraries (if necessary)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Load a custom model
    custom_model = keras.models.load_model(f"Models/MobileFaceNet_model_{latest_version_models}.h5")

    # Convert the custom VGGFace model to TensorFlow Lite
    face_converter = tf.lite.TFLiteConverter.from_keras_model(custom_model)
    face_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    face_converter_tflite = face_converter.convert()

    # Save the converted model to a TFLite file
    with open(new_model_filename, 'wb') as f:
        f.write(face_converter_tflite)

    
if __name__ == "__main__":
    main()
    
    