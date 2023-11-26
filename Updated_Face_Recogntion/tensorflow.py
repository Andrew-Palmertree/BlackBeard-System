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
from cv2.face import LBPHFaceRecognizer
import time

# Set the environment variable to allow duplicated libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load VGGFace model with the 'resnet50' architecture
vggface_resnet = VGGFace('Updated_2_ResNet50_with_Augmentation_pi3.h5')

# Load a custom VGG model from a saved file
custom_vgg_model = keras.models.load_model("Updated_2_ResNet50_with_Augmentation_pi3.h5")

# Define the base learning rate for model compilation
base_learning_rate = 0.0001

# Compile the custom VGG model with specified optimizer, loss, and metrics
custom_vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Create a sequential model by adding softmax layer to the custom VGG model
prob_model = keras.Sequential([custom_vgg_model, tf.keras.layers.Softmax()])

# Initialize LBPHFaceRecognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Load the Haar Cascade classifier for face detection
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# Initialize the video capture object from the default camera (camera index 0)
video_capture = cv2.VideoCapture(0)

# Set camera properties for frame size and frames per second (FPS)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

def detect_bounding_box(vid):
    
    # Cascade classifier for detecting faces
    faces = face_classifier.detectMultiScale(vid, 1.3, 5)
 
    # Cascade Classifier
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        # Region of Interest (ROI) for the detected face
        roi_gray = vid[y:y+h, x:x+w]
        roi_color = vid[y:y+h, x:x+w]
        
        # Resize the detected face to the expected input size (224x224)
        face = cv2.resize(roi_color, (224, 224))
    
        # Expand the dimensions to match the model's input shape
        face = tf.expand_dims(face, axis=0)  

        # Get predictions using the probability model
        predictions = prob_model.predict(face)

        # Print out the predictions
        print("Predictions:")
        for i, pred in enumerate(predictions[0]):
            class_name = f"Class {i}"
            print(f"{class_name}: {pred:.4f}")          
        
    return vid

# Continuous video capture loop
while True:
    result, video_frame = video_capture.read()
    if result is False:
        break

    processed_frame = detect_bounding_box(video_frame)

    cv2.imshow("My Face Detection Project", processed_frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
