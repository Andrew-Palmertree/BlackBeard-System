import cv2
from threading import Thread
# from imutils.video import FPS
import time
import os
from datetime import datetime
import numpy as np
from cv2.face import LBPHFaceRecognizer
import tensorflow as tf
import keras
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras.utils.data_utils import get_file
import keras_vggface.utils
import os.path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize LBPHFaceRecognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Set the environment variable to suppress potential duplicate library loading errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='vggface_resnet_tflite')

# Allocate memory for the model
interpreter.allocate_tensors()

# Get the input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#cascade classifier
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')


def detect_bounding_box(vid):
    
    #cascade classifier
    faces = face_classifier.detectMultiScale(vid, 1.3, 5)
    
    #Cascade Classifier
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        # Region of Interest (ROI) for the detected face
        roi_gray = vid[y:y+h, x:x+w]
        roi_color = vid[y:y+h, x:x+w]
        
        # Resize the face to match the TFLite model input size
        face = cv2.resize(roi_color, (224, 224))
        face = tf.expand_dims(face, axis=0)
        
        # Convert the EagerTensor to a NumPy array
        face = face.numpy()

        # Set the input tensor for the TFLite model
        interpreter.set_tensor(input_details[0]['index'], face.astype('float32'))

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        lite_predictions = interpreter.get_tensor(output_details[0]['index'])
        # Apply softmax to convert raw scores to probabilities
        probabilities = tf.nn.softmax(lite_predictions)


        # Print out the probabilities
        # Initialize variables to track the class with the highest probability and its associated name
        max_prob_class = None
        max_prob_value = 0.0
        name = None

        # Iterate through the probabilities to find the class with the highest probability
        for i, prob in enumerate(probabilities[0]):
            class_name = f"Class {i}"
            print(f"{class_name}: {prob:.4f}")

            if prob >= max_prob_value:
                max_prob_value = prob
                max_prob_class = class_name 

        # Map the class with the highest probability to a name
        if max_prob_class == "Class 0":
            name = "Andrew"
        elif max_prob_class == "Class 1":
            name = "Parbin"
        elif max_prob_class == "Class 2":
            name = "Unknown"
        elif max_prob_class == "Class 3":
            name = "Will"

        # Ensure x, y coordinates are within the frame bounds
        if x >= 0 and y >= 0 and x + w < vid.shape[1] and y + h < vid.shape[0]:
            # Ensure the name text doesn't go beyond the frame width
            if x + w + len(name) * 10 < vid.shape[1]:
                # Display the name associated with the class with the highest probability
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(vid, name, (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)
            else:
                print("Name text exceeds frame width.")
        else:
            print("Bounding box coordinates out of frame bounds.")

        print("Detected Name:", name)

    return vid

#Threading
class PiVideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.camera = cv2.VideoCapture(0)
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
        

def main():
    vs = PiVideoStream().start()
    time.sleep(2.0)
#     fps = FPS().start()

    # Initialize the time variables
    start_time = time.time()
#     interval = 5  # Display FPS every 5 seconds
    face_check_interval = 500  # Check for faces every 50ms
    last_face_check_time = time.time()

    # Initialize recognition mode
    recognition_mode = False

    count = 0

    while True:
        frame = vs.read()

        if frame is not None:
            
            # Display the frame
            cv2.imshow("My Face Detection Project", frame)

            key = cv2.waitKey(1) & 0xFF

            # Check if it's time to check for faces based on the face_check_interval
            current_time = time.time()
            if current_time - last_face_check_time >= face_check_interval / 1000.0:
                # Perform face detection
                processed_frame = detect_bounding_box(frame)
                last_face_check_time = current_time
            else:
                # If not checking for faces, continue displaying the camera feed
                processed_frame = frame

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Check for user input ('q' to exit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

#         fps.update()

#     fps.stop()
#     print("[INFO] Final FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()

