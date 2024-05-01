#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: new_model_mobileFaceNet.py
#
# MobileFaceNet creators: Sheng Chen, Yang Liu, Xiang Gao, Zhen Han
# The script we edited was based on a github user: zye1996 
# Editor of the script to work with BlackBeard system: Andrew Palmertree
#
# Date: March 9, 2024
# Version: 1.0.0
# 
# Requirments:
#     - python 3.9.18
#     - check the requirement.txt file for specific packages
#
# Usage: 
#     used for the AppServer2 script
#
# Description:
#     The new_model_mobileFaceNet script is for the BlackBeard project to be used on the server side to
#     take the train a new model from the user classes located in 'pubfig/train'
#
# Notes:
# This is an edited version based on the following github link:
# https://github.com/zye1996/Mobilefacenet-TF2-coral_tpu/blob/master/train/train.py
#  
# The face recongition model wouldn't have been possible without the the following reseachers
# Sheng Chen, Yang Liu, Xiang Gao, Zhen Han
#
# Find there paper at: https://arxiv.org/abs/1804.07573
#
#############################################################################################


#============================================================================================
# Import the necessary package libraries
#============================================================================================
import os
import tensorflow as tf
import keras
import math
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import SGD

#import the functions in the main_face_rec file
from main_face_rec import countImages


#============================================================================================
# Initialize variables 
#============================================================================================


# Call the countImages function and unpack the returned values
count_trainable_value, folder_name_list, all_folder_name_list = countImages()

# Empty variables that will hold the number of classes from the 'pubfig/train', if resuming training, and the old model path
cls_num = " "
RESUME = " "
LOAD_MODEL_PATH = " " 


#============================================================================================
# Define used functions
#============================================================================================


#############################################################################################
# Get the latest version of the trained MobileFaceNet model
#############################################################################################
def latestModelVersion():
    # used to check if there is a previous trained model to name the new model with the correct version
    no_previous_models = 0
    
    # Get the latest version of the face recongition model
    model_directory = 'Models/'

    # Initialize an empty list for mdoel numbers
    model_numbers = []

    # Loop through all the files in the directories
    for f in os.listdir(model_directory):
        #Check if the file is for face recognition
        if f.startswith('MobileFaceNet_model_'):
            # Extract the model number from the filename
            model_number = int(f.split("_")[-1].split(".h5")[0])
            model_numbers.append(model_number)

    if model_numbers:
        latest_version = max(model_numbers)
    else:
        no_previous_models = 1
        latest_version = 0

    print(f"latest_version: {latest_version}")
    
    if  no_previous_models == 0:
        # Increment the model version
        new_version = latest_version + 1
    else:
        new_version = 0
        
    return new_version


#############################################################################################
# Function to create the label file to hold all user classes names
#############################################################################################
def makeLabelFile(new_version):

    # Get all the user classes names and sorts them numerically
    class_names_list = sorted(folder_name_list)

    labels = '\n'.join(class_names_list)
    
    # File path
    file_path = f'Models\\labels\\labelmap_mobileFaceNet{new_version}.txt'
    
    # Write the content to the file
    with open(file_path, 'w') as f:
        f.write(labels)
        

#############################################################################################
# Function to create the data file
# the content of the file goes as:
# <image_path> <user class #>
#############################################################################################
def createDataFile(new_version):
    
    # Specifies where the images are located
    CLASS_FOLDER = 'pubfig\\resized\\'
    
    # Variable to keep track of the current user class number
    classNum = 0
    
    # File path where the data files are going to be stored
    file_path = f'Models\\DataFiles\\Data_File_{new_version}.txt'
    
    # Dictionary to hold the count of images in each directory
    image_count = {}

    # walks through each image in every folder under CLASS_FOLDER and puts it in the data file
    for root, dirs, files in os.walk(CLASS_FOLDER):
        # Check if the current directory is the current directory and if so exclude it
        if root == CLASS_FOLDER:
            continue
            
        #Filter out images in each of the directories under pubfig/train/
        images = [file for file in files if file.endswith(('png', 'jpg'))]
        
        image_count[os.path.basename(root)] = len(images)
        
        for image in images:
            path = os.path.join(root, image)
            Data_Entrie = (path + " " + str(classNum) + "\n")
            # Write the content to the file
            with open(file_path, 'a') as f:
                f.write(Data_Entrie)
        classNum += 1
        

#############################################################################################
# Class to the contains the MobileFaceNet contents
# Notes:
#    Arguments:
#        inputs: the input embedding vectors
#        n_classes: number of classes
#        s: scaler value (default as 64)
#        m: the margin value (default as 0.5)
#    Returns:
#        the final calculated outputs
#
#
#############################################################################################        
class ArcFace_v2(keras.layers.Layer):

    # Initialize the layer with number of classes, margin (m), and scale (s)
    def __init__(self, n_classes, s=32., m=0.5, **kwargs):
        self.init = keras.initializers.get('glorot_uniform')  # Xavier uniform intializer
        self.n_classes = n_classes
        self.s = s
        self.m = m
        super(ArcFace_v2, self).__init__(**kwargs)

    # Buiils the layer and initialize the weights
    def build(self, input_shape):
        # Build the layer with input shape
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        # Initialize the weight matrix W with shape (input_shape[0][-1], n_classes)
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer=self.init)
        super(ArcFace_v2, self).build(input_shape[0])

    # This is the forward pass of the layer
    # Takes a input data and computest the output layer
    def call(self, inputs, **kwargs):
        # Define the forward pass of the layer
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)

        X, Y = inputs

        X_normed = tf.math.l2_normalize(X, axis=1)  # L2 Normalized X
        W = tf.math.l2_normalize(self.W, axis=0)  # L2 Normalized Weights

        # Calculate cosine similarity between X and W
        cos_theta = tf.keras.backend.dot(X_normed, W)  
        cos_theta2 = tf.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = tf.sqrt(sin_theta2 + tf.keras.backend.epsilon())
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))

        # This condition controls the theta + m should in range [0, pi]
        cond_v = cos_theta - threshold
        cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)

        # Calculate the final output using softmax
        inv_Y = 1. - Y
        s_cos_theta = self.s * cos_theta
        output = tf.nn.softmax((s_cos_theta * inv_Y) + (cos_tm_temp * Y))

        return output

    # returns a dictionary containing the configuration layer so it can be used later
    def get_config(self):
        # Function to serialize layer configuration
        config = {"n_classes": self.n_classes,
                  "s": self.s,
                  "m":self.m
                  }
        base_config = super(ArcFace_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        # Function to compute output shape based on the input shape
        return input_shape[0], self.n_classes

    
#############################################################################################
# Function to load the dataset from a text file containing image paths and labels    
#############################################################################################    
def load_dataset(img_txt_dir, data_root, val_split=0.05):
    image_list = []     # image directory
    label_list = []     # label
    
    # Read image paths and labels from the specified text file
    with open(img_txt_dir) as f:
        img_label_list = f.read().splitlines()
        
    # Extract image directories and labels from the read information    
    for info in img_label_list:
        image_dir, label_name = info.split(' ')
        image_list.append(os.path.join(image_dir))
        label_list.append(int(label_name))

    # Split the dataset into training and testing sets            
    trainX, testX, trainy, testy = train_test_split(image_list, label_list, test_size=val_split)

    return trainX, testX, trainy, testy

#############################################################################################
# Function to preprocess images and labels before training
#############################################################################################
def preprocess(x,y):
    # Read image file
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # Decode JPEG images with 3 channels (RGBA)
    x = tf.image.resize(x, [112, 96])  # Resize images to a fixed size
    x = tf.image.random_crop(x, size=[112, 96, 3])  # Randomly crop images
    x = tf.image.random_flip_left_right(x)  # Randomly flip images horizontally
    x = tf.image.random_brightness(x, max_delta=0.2)  # Randomly adjust brightness
    x = tf.image.random_contrast(x, lower=0.5, upper=1.5)  # Randomly adjust contrast
    x = tf.image.random_saturation(x, lower=0.5, upper=1.5)  # Randomly adjust saturation
    x = tf.image.random_hue(x, max_delta=0.2)  # Randomly adjust hue

    # Normalize image values to the range [-1, 1]
    x = (tf.cast(x, dtype=tf.float32) - 127.5) / 128.0
    
    # Convert label to one-hot encoding
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=cls_num)

    # Return preprocessed image and label
    if RESUME:
        return (x, y), y
    else:
        return x, y



#############################################################################################
# Function to construct the MobileFaceNet model for training
#############################################################################################
def mobilefacenet_train(softmax=False):   

    # Load pre-trained model
    model = tf.keras.models.load_model(LOAD_MODEL_PATH)
    
    # Extract input and output tensors from the pre-trained model
    inputs = model.input
    x = model.output
    
    # Define a new input tensor for target labels
    y = tf.keras.layers.Input(shape=(cls_num,), name="target")
    
    # Connect the MobileFaceNet output to ArcFace layer
    outputs = ArcFace_v2(n_classes=cls_num)((x, y))

    return tf.keras.models.Model([inputs, y], outputs)


#############################################################################################
# Function to load the old model and train a new model
#############################################################################################
def main():
    global cls_num, RESUME, LOAD_MODEL_PATH
    
    # Functions to get the lastest model version and create the new label and data file
    new_version = latestModelVersion()   # Get the latest model version
    makeLabelFile(new_version)           # Create a new label file
    createDataFile(new_version)          # Create a new data file
    
    # If there are no previous versions, then start training from scratch
    if new_version == 0:
        print("Creating a new model starting at 0")
        # Configuration for starting from scratch
        LOAD_MODEL = 0
        LOAD_MODEL_PATH = r"C:\Users\andre\CPE_4093\Senior_project\Server\inference_model_all.h5"
        RESUME = True
        
        # load dataset
        data_root = r"C:\Users\andre\CPE_4093\Senior_project\Server"
        img_txt_dir = os.path.join(data_root, 'CASIA-WebFace-112X96.txt')
        data_file = 'CASIA-WebFace-112X96.txt'
        
    # If there is a previous model, then train on top of that
    else:
        print("Grabbing last model version")
        # Configuration for continuing from the previous version
        LOAD_MODEL = 0
        previous_model = new_version - 1
        LOAD_MODEL_PATH = rf"C:\Users\andre\CPE_4093\Senior_project\Server\Models\MobileFaceNet_model_{previous_model}.h5"
        RESUME = True
        
        # load dataset
        data_root = rf"C:\Users\andre\CPE_4093\Senior_project\Server\Models\DataFiles"
        img_txt_dir = os.path.join(data_root, f'Data_File_{new_version}.txt')
        data_file = f'Data_File_{new_version}.txt'
        
    
    # get data slices
    train_image, val_image, train_label, val_lable = load_dataset(img_txt_dir, data_root)

    # get class number
    cls_num = len(np.unique(train_label))

    # Set batch size
    batchsz = 8
    
    # Construct train dataset
    db_train = tf.data.Dataset.from_tensor_slices((train_image, train_label)) 
    db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
    
    # Construct validation dataset
    db_val = tf.data.Dataset.from_tensor_slices((val_image, val_lable))
    db_val = db_val.shuffle(1000).map(preprocess).batch(batchsz)

    # Load or create model
    if LOAD_MODEL != 0:
        model = keras.models.load_model(LOAD_MODEL_PATH)
    else:
        model = mobilefacenet_train(softmax=False)
        print(model.summary())


    # Compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # Train model
    history = model.fit(db_train, validation_data=db_val, validation_freq=1, epochs=50, initial_epoch=34)

    # Save inference model save
    inference_model = keras.models.Model(inputs=model.input[0], outputs=model.layers[-3].output)
    inference_model.save(rf'C:\Users\andre\CPE_4093\Senior_project\Server\Models\MobileFaceNet_model_{new_version}.h5')
    
    

if __name__ == '__main__':
    main()