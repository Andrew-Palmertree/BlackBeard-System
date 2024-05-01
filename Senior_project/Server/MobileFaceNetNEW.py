#############################################################################################
# Project Name: BlackBeard
# Created by: Parbin Darji, Will Fortenberry, Andrew Palmertree, Imanol  Perales, Justin Romanowski
#
# Script name: MobileFaceNetNEW.py
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
#     The MobileFaceNetNEW script is for the BlackBeard project to be used to manually
#     train a new model from the user classes located in 'pubfig/train'
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
from tensorflow.keras.regularizers import l2

#import the functions in the main_face_rec file
from main_face_rec import countImages


#============================================================================================
# Define used functions
#============================================================================================


# Call the countImages function and unpack the returned values
count_trainable_value, folder_name_list, all_folder_name_list = countImages()


#############################################################################################
# Get the latest version of the trained MobileFaceNet model
#############################################################################################
def latestModelVersion():
    # used to check if there is a previous trained model to name the new model with the correct version
    no_previous_models = 0
    
    # Get the latest version of the face recongition model
    model_directory = 'MobileFaceNetNew/'

    # Initialize an empty list for model numbers
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
    CLASS_FOLDER = 'pubfig\\train\\'
    
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
# Weight decay setting
#############################################################################################     
#1e-5 adjust as needed
weight_decay = 4e-5   


#############################################################################################
# Bottleneck block used in MobileFaceNet (Filter information)
############################################################################################# 
def bottleneck(x_in, d_in, d_out, stride, depth_multiplier):
 # Decide whether there would be a short cut
    if stride == 1 and d_in == d_out:
        connect = True
    else:
        connect = False

    # Point-wise layers
    x = keras.layers.Conv2D(d_in * depth_multiplier, kernel_size=1, strides=1, padding='VALID',
                            use_bias=False, kernel_regularizer=l2(weight_decay))(x_in)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Depth-wise layers
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)  # Manually padding
    x = keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='VALID',
                                     use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Point-wise layers linear
    x = keras.layers.Conv2D(d_out, kernel_size=1, strides=1, padding='VALID',
                            use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = keras.layers.BatchNormalization()(x)

    if connect:
        return keras.layers.Add()([x, x_in])  # Shortcut connection
    else:
        return x
    
    
#############################################################################################
#Convolutional block used in MobileFaceNet
#############################################################################################      
def conv_block(x_in, d_in, d_out, kernel_size, stride, padding, depthwise=False, linear=False):

    # Padding if needed
    if padding != 0:
        x = keras.layers.ZeroPadding2D(padding=(padding, padding))(x_in)
    else:
        x = x_in

    # Convolutional layer (depthwise or standard)
    if depthwise:
        x = keras.layers.DepthwiseConv2D(kernel_size, strides=stride, padding='VALID',
                                         use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    else:
        x = keras.layers.Conv2D(d_out, kernel_size, strides=stride, padding='VALID',
                                use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Batch normalization
    x = keras.layers.BatchNormalization()(x)

    # Activation function (PReLU or linear)
    if not linear:
        return keras.layers.PReLU(shared_axes=[1, 2])(x)
    else:
        return x  
    
    
#############################################################################################
# Each sublist within Mobilefacenet_bottleneck_setting represents a block
# 't' Expansion factor
# 'c' Number of output channels of the 3x3 convolutional layer 
# 'n' Number of times the bottleneck block is repeated
# 's' Stride of the first 3x3 convolutional layer inside the bottleneck block
#############################################################################################     
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

#############################################################################################
# Function to create the MobileFaceNet model architectur
#############################################################################################        
def mobilefacenet(x_in, inplanes=64, setting=Mobilefacenet_bottleneck_setting):
    # Initial convolutional block
    x = conv_block(x_in, d_in=3, d_out=64, kernel_size=3, stride=2, padding=1)

    # Depthwise convolutional block
    x = conv_block(x, d_in=64, d_out=64, kernel_size=3, stride=1, padding=1, depthwise=True)
    
    # Loop through MobileFaceNet bottleneck settings
    for t, c, n, s in setting:
        for i in range(n):
            if i == 0:
                # First bottleneck block in a sequence
                x = bottleneck(x, inplanes, c, s, t)
            else:
                # Subsequent bottleneck blocks in a sequence
                x = bottleneck(x, inplanes, c, 1, t)
            inplanes = c
    
    # Convolutional block to reduce dimensions
    x = conv_block(x, d_in=128, d_out=512, kernel_size=1, stride=1, padding=0)
    
    # Depthwise convolutional block with linear activation
    x = conv_block(x, d_in=512, d_out=512, kernel_size=(7, 6), stride=1, padding=0,
                   depthwise=True, linear=True)
    
    # Convolutional block with linear activation
    x = conv_block(x, d_in=512, d_out=128, kernel_size=1, stride=1, padding=0,
                   linear=True)
    
    # Flatten layer
    x = keras.layers.Flatten()(x)

    return x     


# Configuration for continuing from the previous version
LOAD_MODEL = 0
LOAD_MODEL_PATH = r"MobileFaceNetNew\inference_model_all.h5"
RESUME = True


# load dataset
data_root = r"MobileFaceNetNew"
img_txt_dir = os.path.join(data_root, 'CASIA-WebFace-300X300.txt')


#############################################################################################
# Function to load the dataset from a text file containing image paths and labels    
############################################################################################# 
def load_dataset(val_split=0.05):
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
    x = tf.image.resize(x, [300, 300])  # Resize images to a fixed size
    x = tf.image.random_crop(x, size=[300, 300, 3])  # Randomly crop images
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

# get data slices
train_image, val_image, train_label, val_lable = load_dataset()

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


#############################################################################################
# Function to construct the MobileFaceNet model for training
#############################################################################################
def mobilefacenet_train(softmax=False):   
    
    # CONFIG
    LOAD_MODEL = 0
    LOAD_MODEL_PATH = "inference_model_all.h5"
    RESUME = False

    if RESUME:
        model = tf.keras.models.load_model(LOAD_MODEL_PATH)
        inputs = model.input
        x = model.output
    else:
        x = inputs = tf.keras.layers.Input(shape=(300, 300, 3))
        x = mobilefacenet(x)
        
    y = tf.keras.layers.Input(shape=(cls_num,), name="target")
    outputs = ArcFace_v2(n_classes=cls_num)((x, y))

    return tf.keras.models.Model([inputs, y], outputs)


#############################################################################################
# Function to load the old model and train a new model
#############################################################################################
def main():
    # Functions to get the lastest model version and create the new label and data file
#     new_version = latestModelVersion()   # Get the latest model version
#     makeLabelFile(new_version)           # Create a new label file
#     createDataFile(new_version)          # Create a new data file
    
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
    if RESUME:
        print("wrong")
        inference_model = keras.models.Model(inputs=model.input[0], outputs=model.layers[-3].output)
        inference_model.save(rf'MobileFaceNetNew\MobileFaceNet_model_2.h5')
    else:
        print("correct")
        inference_model = keras.models.Model(inputs=model.input[0], outputs=model.layers[-3].output)
        inference_model.save(r'MobileFaceNetNew\MobileFaceNet_model_testing.h5')


if __name__ == '__main__':
    main()