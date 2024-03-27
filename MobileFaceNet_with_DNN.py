#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: MobileFaceNet_with_DNN.py
#
# Date: March 9, 2024
# Version: 1.0.0
# 
# Requirments:
#     - python 3.9.2
#     - check the requirement.txt file for specific packages
#
# Description:
#     The MobileFaceNet_with_DNN script is for the BlackBeard project to be used on the microcontroller side to
#     detect faces and objects.
#
#      Object list: amazon logo, amazon delivery uniform, delivery car, fedex logo,
#                   fedex uniform, package, person, safety vest, ups logo, ups uniform
#                   usps logo, and usps delivery uniform
#
# Notes:
# The object recognition runs on the Google Corel TPU USB accelerator
# The face recognition runs on the microcontroller CPU for better accuracy 
#
#############################################################################################


#============================================================================================
# Import the necessary package libraries
#============================================================================================
import os
import argparse

import cv2
import numpy as np
import sys
import time
from threading import Thread, Event
import importlib.util
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
import glob
import requests
import subprocess
import json


#============================================================================================
# Initialize variables 
#============================================================================================


latest_version_converted = -1  # Shared variable to store the latest version
update_event = Event()  # Event to signal when to update the latest version
current_version = -1 # Used to keep track if there is a new trained tflite file available

# Load the pre-trained model and configuration for DNN face extraction
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Get current working directory
current_directory = os.getcwd()

# IP address of server
# IP_ADDRESS = '192.168.2.203'
IP_ADDRESS = '10.101.169.96'

# Used to determine to send face and object prediction data to the server
send_live_image = False

# Used to store the object and face recognition data 
face_predictions = []
object_predictions = []

frame_rate = 5  # Target frame rate, 5 frames per second
previous_time = time.time() # Used for checking the time for upload live feed to server for the app

#============================================================================================
# Define used functions
#============================================================================================


#############################################################################################
# Preprocesses an image for input to a neural network
#############################################################################################
def preprocess(img):

    # Normalize pixel values to [-1, 1]
    img = (img.astype("float32") - 127.5) / 128.0
    # Add batch dimension of batch size of 1
    img = np.expand_dims(img, axis=0)
    return img


#############################################################################################
# Function to load the TFLite model capable of running on the Edge TPU
#############################################################################################
def load_tflite_model(model_path):
    
    interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    interpreter.allocate_tensors()
    return interpreter


#############################################################################################
# Function to detect faces and draw bounding boxes around the faces with labels
#############################################################################################
def detect_bounding_box(vid, interpreter):
    
    # Get the height and width of the frame
    h, w = vid.shape[:2]

    # Prepare the image for input to the DDN face extraction
    blob = cv2.dnn.blobFromImage(cv2.resize(vid, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

    # Set the input blob for the neural network
    net.setInput(blob)

    # Perform forward pass inference to detect faces
    detections = net.forward()

    # Draw bounding boxes around the detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(vid, (x, y), (x1, y1), (0, 0, 255), 2)

            # Region of Interest (ROI) for the detected face
            roi_color = vid[y:y1, x:x1]

            # Resize the detected face to the expected input size (96x112) the MobileFaceNet model
            face = cv2.resize(roi_color, (96, 112))

            # Preprocess the face image
            face = preprocess(face)

            # Run inference on the preprocessed face
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

        
            interpreter.set_tensor(input_details[0]['index'], face)
            interpreter.invoke()
            embedding_yzy = interpreter.get_tensor(output_details[0]['index'])

            embedding_yzy = embedding_yzy / np.expand_dims(np.sqrt(np.sum(np.power(embedding_yzy, 2), 1)), 1)
            
            prediction = 0
            
            # Directory where the Data_File_#.txt files will be stored 
            data_file = f"Models/dataBase/Data_File_{latest_version_converted}.txt"
            DATA_FILE_LOCATION = os.path.join(current_directory, data_file)
            
            # Latest Data File used to compare with the live image to determine if it's a known user in frame
            file_data = DATA_FILE_LOCATION
            print("Latest version: " + str(latest_version_converted))
            
            data = np.loadtxt(file_data)
            
            num_entries = data.shape[0]
            
            list_t = []
            
            # Directory where the labelmap_mobileFaceNet#.txt files will be stored 
            label_file = f"Models/Edge/labelmap_mobileFaceNet{latest_version_converted}.txt"
            LABEL_FILE_LOCATION = os.path.join(current_directory, label_file)
            
            # Label map file for known user classes names   
            labelmap_file = LABEL_FILE_LOCATION

            # Read the label map file
            class_names = {}
            with open(labelmap_file, 'r') as file:
                for line in file:
                    # Split each line by '-' to extract the class number and name
                    class_number, class_name = line.strip().split('-')
                    class_names[int(class_number)] = class_name

            # Make a prediction if the detected face is a known user
            for i in range(num_entries):
                entry = data[i]
                predictions = np.sum(np.multiply(embedding_yzy, entry), 1)
                list_t.append(predictions)
                if prediction < predictions and predictions >= 0.6 and class_names != "Unknown":
                    prediction = predictions
                    className = class_names[i+1]
                elif i == max(range(num_entries)) and prediction <0.6:
                    prediction = predictions 
                    className = "Unknown"
            
            # Used in grabing the prediction either from a nparray, list, or tulpe and convert it into a float  
            if isinstance(prediction, np.ndarray):
                if prediction.size == 1:
                    prediction = prediction.item()
            prediction = float(prediction)    
            
            if (className == "Unknown"):
                face_predictions.append(className)
                
            else:
                name_and_prediction = f"{className}: {prediction*100:.2f}%"
                face_predictions.append(name_and_prediction)
                
            print(prediction) # Give the prediction percentage of how likely the detected face is a known user
            print(className) # This will print the user name of the predicted class (Example: Justin, Andrew, Will, Imanol, or Parbin)
            print(" ")
            font_scale = 2  # Increase this value to make the text bigger
            thickness = 3  # Increase this value to make the text bolder
            text_width, text_height = cv2.getTextSize(className, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(vid, className, (x + (x1 - x - text_width) // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    return vid


#############################################################################################
# Function to check if there is a newer face recognition model available
#############################################################################################
def getLatestConverted():
    global latest_version_converted  # Use global variable to store latest version
    # Directory where the models are stored
    models_directory_converted = 'Models/Edge'
    
    # Initialize an empty list for model numbers
    model_numbers_converted = []
    
    # Loop through all files in the directory
    for f in os.listdir(models_directory_converted):
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
    
    #Get the latest version of the converted model
    old_model_edgetpu_filename = f"Converted_mobileFaceNet_model_{latest_version_converted}_edgetpu.tflite"
    
    return old_model_edgetpu_filename, latest_version_converted
    
    update_event.set()
    

    
#############################################################################################
# Class used threading to capture the video
#############################################################################################
class VideoStream:
    ##
    ## CHECK FRAME SIZE
    ##
    def __init__(self, resolution=(773, 580), framerate=30):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.camera.set(3, resolution[0])
        self.camera.set(4, resolution[1])
        self.camera.set(5, framerate)
        self.frame = None
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.camera.read()
            if not ret:
                break
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        

#############################################################################################
# Function to load the face and object recognition models
#############################################################################################
def models():
    

    # Directory where the Converted_mobileFaceNet_model_#_edgetpu.tflite files will be stored 
    face_model_file = f"Models/Edge/Converted_mobileFaceNet_model_{latest_version_converted}_edgetpu.tflite"
    FACE_MODEL_FILE_LOCATION = os.path.join(current_directory, face_model_file)
    
    # Load the TFLite model  for the face recognition
    model_path = FACE_MODEL_FILE_LOCATION
    #
    ###Test with different dimensions###
    #
    resW, resH = 640,480
    imW, imH = int(resW), int(resH)



    # Directory where the object recognition files will be stored 
    object_model_file = f"custom_model_lite2"
    OBJECT_MODEL_FILE_LOCATION = os.path.join(current_directory, object_model_file)
    
    ## object recognition model
    MODEL_NAME2 = OBJECT_MODEL_FILE_LOCATION
    GRAPH_NAME2 = 'edgetpu.tflite'
    LABELMAP_NAME2 = 'labelmap.txt'
    min_conf_threshold = float(0.5)

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT2 = os.path.join(CWD_PATH,MODEL_NAME2,GRAPH_NAME2)

    # Path to label map file for object recognition
    PATH_TO_LABELS2 = os.path.join(CWD_PATH,MODEL_NAME2,LABELMAP_NAME2)
        
    # Load the label map for object recognition
    with open(PATH_TO_LABELS2, 'r') as f:
        labels2 = [line.strip() for line in f.readlines()]
                                  
    # Load interpreter for face recognition
    interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    
    interpreter.allocate_tensors()
    
    # Load interpreter for object recognition
    interpreter2 = Interpreter(model_path=PATH_TO_CKPT2,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    interpreter2.allocate_tensors()
    
    input_details2 = interpreter2.get_input_details()
    output_details2 = interpreter2.get_output_details()
    height2 = input_details2[0]['shape'][1]
    width2 = input_details2[0]['shape'][2]
    
    floating_model=False

    input_mean = 127.5
    input_std = 127.5

    outname2 = output_details2[0]['name']
    
        
    if ('StatefulPartitionedCall' in outname2): # This is a TF2 model
        boxes_idx2, classes_idx2, scores_idx2 = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx2, classes_idx2, scores_idx2 = 0, 1, 2    
    
    return imW, imH, width2, height2, interpreter2, input_details2, output_details2, boxes_idx2, classes_idx2, scores_idx2, min_conf_threshold, labels2, interpreter
    
    
#############################################################################################
# Function tto determine if wifi is on
# This is used to determine if the microcontroller can send data to the server
#############################################################################################
def has_wifi_connection():
    try:
        output = subprocess.check_output(['nmcli', '-t', '-f', 'STATE', 'g']).decode('utf-8').strip()
        return output == 'connected'
    except subprocess.CalledProcessError:
        return False
    
    
#############################################################################################
# Function to send data to the server including vidoe frames, class names, and predictions
#############################################################################################
def send_to_server(frame, face_predictions, object_predictions):
    if send_live_image == True:
        if not has_wifi_connection():
            print("Not sending the frame. Either WiFi is not connected or sending is disabled.")
            return
        
        # Convert frame (NumPy array) to bytes in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Assuming 'frame' is a video frame and 'predictions' is a dictionary of your predictions
        url = f'http://{IP_ADDRESS}:5353/video_upload'
        
        # Prepare files and data for the request
        files = {'video_frame': ('frame.jpg', frame_bytes, 'image/jpeg')}
        data = {'face_predictions': json.dumps(face_predictions),
                'object_predictions': json.dumps(object_predictions)}
        
        # Send POST request
        response = requests.post(url, files=files, data=data)


#############################################################################################
# Function to see if the microcontroller needs to send live video feed to server
#############################################################################################
def get_video_status():
    global send_live_image
    # Construct the URL for the GET request
    url = f'http://{IP_ADDRESS}:5353/get_video_active'
    
    # Make the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.text == '0':
        send_live_image = False
    else:
        send_live_image = True


#############################################################################################
# Function to continuous run the face and object recognition until
# user stops it by pressing 'q'
#############################################################################################
def main():
    global latest_version_converted, previous_time, face_predictions, object_predictions
   
    
    # Start a thread for getLatestConverted() function to determine if a new version of the face recognition model is available
    get_latest_thread = Thread(target=getLatestConverted)
    get_latest_thread.start()
    
    last_update_time = time.time() # Used to check for the latest new version of the face recognition model
    last_update_time2 = time.time() # Used to check for faces
    last_update_time3 = time.time() # Used to check if live video should be active

    # Grabs the latest face recognition model
    old_model_edgetpu_filename, latest_version_converted = getLatestConverted()
    
    # Grabs the details from the object and face recogntion models
    imW, imH, width2, height2, interpreter2, input_details2, output_details2, boxes_idx2, classes_idx2, scores_idx2, min_conf_threshold, labels2, interpreter = models()
   
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)
    
    check_count = 0

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:
        
        current_version = latest_version_converted 
        # Check if 10 seconds have passed since the last update
        if time.time() - last_update_time >= 10:
                check_count += 1
                # If 10 seconds has passed then check for a newer version of the face recognition model
                old_model_edgetpu_filename, latest_version_converted = getLatestConverted()
                print("Latest version:", latest_version_converted)
                # Grab the new model if available. Otherwise, let the user known that they have the latest version already
                if latest_version_converted != current_version:
                    imW, imH, width2, height2, interpreter2, input_details2, output_details2, boxes_idx2, classes_idx2, scores_idx2, min_conf_threshold, labels2, interpreter  = models()
                    print("NEW MODEL IS ACTIVE")
                    check_count = 0
                else:
                    print(f"No new version of the face recognition model interatipn: {check_count}")
                    print(" ")
                last_update_time = time.time()
                
        # If 0.5 seconds has passed then check if video status is active or not        
        if time.time() - last_update_time3 >= 2:
            get_video_status()
            last_update_time3 = time.time()
            

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()
        
        if frame1 is not None:
            current_time = time.time()
            if (current_time - previous_time) >= 1/frame_rate:
                # get_video_status
                
                # It's time to send the next frame
                send_to_server(frame1, face_predictions, object_predictions)
                
                # Update the previous time
                previous_time = current_time 
            
            face_predictions = []
            object_predictions = []     

            # Face recognition part
            
            if time.time() - last_update_time2 >= 0.01:
                last_update_time2 = time.time()
                processed_frame = detect_bounding_box(frame1, interpreter)     
            

            # Object detection  part
            
            frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width2, height2))
            input_data2 = np.expand_dims(frame_resized, axis=0)
            
            interpreter2.set_tensor(input_details2[0]['index'],input_data2)
            interpreter2.invoke()
            
            # Retrieve detection results
            boxes = interpreter2.get_tensor(output_details2[boxes_idx2]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter2.get_tensor(output_details2[classes_idx2]['index'])[0] # Class index of detected objects
            scores = interpreter2.get_tensor(output_details2[scores_idx2]['index'])[0] # Confidence of detected objects
                        
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame1, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels2[int(classes[i])] # Look up object name from "labels" array using class index
                    label2 = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    object_predictions.append(label2) # send predictions to the app
                    labelSize, baseLine = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame1, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame1, label2, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            

            # Draw framerate in corner of frame
            cv2.putText(frame1,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame1)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1
        

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()


if __name__ == "__main__":
    main()

