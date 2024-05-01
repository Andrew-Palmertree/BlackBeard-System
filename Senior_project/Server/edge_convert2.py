#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: edge_convert2.py
# Author of the script: Andrew Palmertree
# Date: March 9, 2024
# Version: 2.0.0
# 
# Requirments:
#     - python 3.9.18
#     - check the requirement.txt file for specific packages
#     - We are using a windows 11 computer as the server but running the conversion script 
#       we need an ubuntu server. We used Windows Subsystem for Linux (WSL) and installed
#       Ubuntu on it and followed the instruction on to install the edgetpu_compiler command:
#       https://coral.ai/docs/edgetpu/compiler/#system-requirements 
#
# Usage: 
#     used for the AppServer2 script
#
# Description:
#     The edge_convert2 script is for the BlackBeard project to be used on the server side to
#     convert the .tflite model create by the converter2 script to a edgetpu.tflite file.
#
#
#
# Notes:
# The edgeput.tflite file is used so the model can run while the Google Corel TPU is running 
# the object recognition model
#
# Website to get more information on Google Corel TPU used: 
# https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux
#
#############################################################################################


#============================================================================================
# Import the necessary package libraries
#============================================================================================
import paramiko
from pathlib import Path
import os
import shutil
import subprocess


#============================================================================================
# Initialize variables 
#============================================================================================

# Get current working directory
current_directory = os.getcwd()

# Sets the Converted .tflite model directory path
convert_folder = 'Models/Converted/'
CONVERT_FOLDER = os.path.join(current_directory, convert_folder)

# Sets the Edge edgetpu.tflite model directory path
convert_edge_folder = 'Models/Edge/'
CONVERT_EDGE_FOLDER = os.path.join(current_directory, convert_edge_folder)

# Local WSL Ubuntu server directory model location from windows 
ubuntu_model_dir = r'\\wsl.localhost\Ubuntu\home\andrew\Models'

# Ubuntu server file location of model location from Ubunto perspective
ubuntu_model_folder = rf'/home/andrew/Models/'

# Ubuntu edgetpu .tflite files location
ubuntu_edge_folder = rf'/home/andrew/Models/Edge/'

# Ubuntu home folder
ubuntu_home_folder = rf'/home/andrew/'

# Windows server location from the perspective of the ubuntu server
windows_edge_folder = '/mnt/c/Users/andre/CPE_4093/Senior_project/Server/Models/Edge/'

# Ubuntu server IP address, name, and password to ssh into it
Ubuntu_IP = '172.28.57.59'
Ubuntu_NAME = 'andrew'
Ubuntu_PASSWORD = 'kNLC7]B}$W6@3s^8U+wG<KJ(9hD>X?'


#============================================================================================
# Define used functions
#============================================================================================


#############################################################################################
# Function to get the latest version of the trained models
#############################################################################################
def getLatestConverted():
    
    # Directory where the models are stored
    models_directory_converted = CONVERT_FOLDER
    
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
    if model_numbers_converted:
        latest_version_converted = max(model_numbers_converted)
    else:
        latest_version_converted = 0

    print(f"latest_version of normal tflite: {latest_version_converted}")
    
    #Get the latest version of the converted model
    old_model_filename = f"Converted_mobileFaceNet_model_{latest_version_converted}.tflite"

    return old_model_filename
    

#############################################################################################
# Function to convert .tflite to edgetpu.tflite to run the model with the edge tpu 
#############################################################################################    
def main():    
    
    # Get the latest version of the model
    old_model_filename = getLatestConverted()
    
    # Source of the old model
    old_model_file = f"{old_model_filename}"
    source_path = os.path.join(CONVERT_FOLDER, old_model_file)
    
    # Local Ubuntu server model directory location from windows 
    destination_path_dir = ubuntu_model_dir
    
    # Ubuntu server model directory location from Ubuntu perspective
    old_model_ubuntu = f"{old_model_filename}"
    destination_path_file = os.path.join(ubuntu_model_folder, old_model_ubuntu)
    
    # Make sure the file doesn't already exist otherwise it won't copy over
    if Path(fr'{destination_path_dir}\{old_model_filename}').is_file():
        print("file already exist")
    else:
        shutil.copy(source_path, destination_path_dir)
        print("File doesn't exist. Copying over file.")  
    
    # Finds the latest converted model version name 
    edgeLog = old_model_filename.split('.tflite')[0]
    
    # Create new file name for edgetpu .tflite file
    edgetpu_file =f"{edgeLog}_edgetpu.tflite"
    edgeFILE = os.path.join(CONVERT_EDGE_FOLDER, edgetpu_file)
    
    print("Final path:", edgeFILE) 
    if os.path.exists(edgeFILE):
        print("File exists!")
    else:
        print("File does not exist.")

    # Connect to the local WSL Ubuntu server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(Ubuntu_IP, username=Ubuntu_NAME, password=Ubuntu_PASSWORD)
    
    # Finds the latest converted model version name
    edgeLog = old_model_filename.split('.tflite')[0]
    
    # Determines if the new edgetpu model exist
    command = 'test -f {}/{}_edgetpu.tflite && echo 1 || echo 0'.format(ubuntu_edge_folder, edgeLog)
    stdin, stdout, stderr = ssh.exec_command(command)
    edge_exist = int(stdout.read().decode().strip())
    
    # if the new edgetpu model deoesn't exist then create it
    if edge_exist != 1:
        
        # Create the the new edgetpu.tflite file and edgetpu.log file
        stdin, stdout, stderr = ssh.exec_command('edgetpu_compiler ' + destination_path_file)
        print(stdout.readlines())

        # Determines if the edgetpu.log exist
        command1 = 'test -f {}/{}_edgetpu.log && echo 1 || echo 0'.format(ubuntu_home_folder, edgeLog)
        stdin, stdout, stderr = ssh.exec_command(command1)
        edge = int(stdout.read().decode().strip())

        # If the edgetpu.tflite exist, then delete it
        # This file is not necessary and create a lot of clutter in the folder over time
        if edge == 1:
            command2 = 'rm -rf {}/{}_edgetpu.log'.format(ubuntu_home_folder, edgeLog)
            stdin, stdout, stderr = ssh.exec_command(command2)
            print(stdout.read().decode().strip())
        else:
            print(f"The {edgeLog}_edgetpu.log file doesn't exsit")


        # Moves the edgetpu.tflite file from the home folder to the 'Models/Edge' 
        command3 = 'mv {}/{}_edgetpu.tflite {}/Models/Edge'.format(ubuntu_home_folder, edgeLog, ubuntu_home_folder)
        stdin, stdout, stderr = ssh.exec_command(command3)
        print(stdout.read().decode().strip())        

        # Move the edgetpu.tflite from the Ubuntu server to the Windows server
        command4 = 'cp {}/{}_edgetpu.tflite {}'.format(ubuntu_edge_folder, edgeLog, windows_edge_folder)
        stdin, stdout, stderr = ssh.exec_command(command4)
        print(stderr.read().decode().strip())
        
        print("Disconnecting from Ubuntu server")
    
    else:
        print("Already converted the latest model")

    # Disconnect from the Ubuntu server    
    ssh.close()
    
print("EDGETPU finished converting")
    

if __name__ == "__main__":
    main()
    