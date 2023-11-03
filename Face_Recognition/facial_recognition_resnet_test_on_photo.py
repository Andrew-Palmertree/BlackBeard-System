import tensorflow as tf
import keras
import keras_vggface
from keras_vggface.vggface import VGGFace
import mtcnn
import numpy as np
import matplotlib as mpl
from matplotlib.image import imread
import matplotlib.pyplot as plt
from keras.utils.data_utils import get_file
import keras_vggface.utils
import PIL
import os
import os.path
import cv2
import os

# Set the environment variable to allow duplicated libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the VGGFace model (resnet50)
vggface_resnet = VGGFace('resnet50')

# Load a custom model from a saved file
custom_vgg_model = keras.models.load_model("test2.h5")

# Load and preprocess an image (scalia_photo) to match the model's input shape
scalia_photo = imread('public_images/2.jpg')

# Perform face detection using MTCNN (you can keep this part as it is)
face_detection = mtcnn.MTCNN()
face_roi = face_detection.detect_faces(scalia_photo)

# Extract the face region from the detected faces and resize it to the model's input size
x1, y1, width, height = face_roi[0]['box']
x2, y2 = x1 + width, y1 + height
face = scalia_photo[y1:y2, x1:x2]
face = cv2.resize(face, (224, 224))

# Show the face image
plt.imshow(face)
plt.show()

# Set a base learning rate and compile the custom model
base_learning_rate = 0.0001
custom_vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

# Create a probability model by adding a Softmax layer
prob_model = keras.Sequential([custom_vgg_model, tf.keras.layers.Softmax()])

# Prepare the face image for prediction
face = tf.expand_dims(face, axis=0)  

# Make predictions using the probability model
predictions = prob_model.predict(face)

# Print out the predictions
print(predictions)
print("Predictions:")
for i, pred in enumerate(predictions[0]):
    class_name = f"Class {i}"
    print(f"{class_name}: {pred:.4f}")
