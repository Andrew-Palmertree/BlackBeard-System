# I commented the packages I think you won't need but left them in just incase
import tensorflow as tf
import keras
import keras_vggface
from keras_vggface.vggface import VGGFace
#import mtcnn
import numpy as np
#import matplotlib as mpl
#from matplotlib.image import imread
#import matplotlib.pyplot as plt
from keras.utils.data_utils import get_file
import keras_vggface.utils
#import PIL
import os
import os.path

# Set the environment variable to allow duplicated libraries (if necessary)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the VGGFace model with pre-trained weights
vggface_resnet = VGGFace('test4.h5')

# Load a custom model
custom_vgg_model = keras.models.load_model("test4.h5")

# Define the base learning rate
base_learning_rate = 0.0001

# Compile the custom model with specific optimizer, loss, and metrics
custom_vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Create a sequential model that includes the custom model and a softmax layer
prob_model = keras.Sequential([custom_vgg_model, tf.keras.layers.Softmax()])

# Convert the custom VGGFace model to TensorFlow Lite
vggface_resnet_converter = tf.lite.TFLiteConverter.from_keras_model(custom_vgg_model)
vggface_resnet_converter.optimizations = [tf.lite.Optimize.DEFAULT]
vggface_resnet_tflite = vggface_resnet_converter.convert()

# Save the converted model to a TFLite file
with open('vggface_senet50_tflite', 'wb') as f:
    f.write(vggface_resnet_tflite)
