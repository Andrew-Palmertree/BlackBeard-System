import cv2
from threading import Thread
from imutils.video import FPS
import time
import os
from datetime import datetime
import numpy as np
from cv2.face import LBPHFaceRecognizer
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize LBPHFaceRecognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Cascade classifier for face detection
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

recognizer_dir = os.path.join(BASE_DIR, "recognizer")

# Load the trained recognizer model
recognizer.read(recognizer_dir+ "/trainnerTest_360.yml")

# Load label mappings from a pickle file
labels = {"person_name": 1}
with open("pickles/face-labels_360.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

def detect_bounding_box(vid):
     # Convert the frame to grayscale for face detection
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the cascade classifier
    faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)
    
    # Draw bounding boxes around detected faces and perform recognition
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        # Region of Interest (ROI) for the detected face
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = vid[y:y+h, x:x+w] 
        
        # Perform face recognition and display the name if recognized
        id_, conf = recognizer.predict(roi_gray)
        print(id_, conf)
        if conf>=4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(vid, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            
    return vid

# Define a video stream class for capturing frames
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
        
# Main function for running the application
def main():
    vs = PiVideoStream().start()
    time.sleep(2.0)
    fps = FPS().start()

    # Initialize the time variables
    start_time = time.time()
    interval = 5  # Display FPS every 5 seconds
    
    # Initialize recognition mode
    recognition_mode = False
    
    count = 0

    while True:
        frame = vs.read()

        if frame is not None:
            # Display the frame
            #cv2.imshow("Frame", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Process the frame by detecting faces and performing recognition
            processed_frame = detect_bounding_box(frame)

            cv2.imshow("My Face Detection Project", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Check for user input ('q' to exit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        fps.update()


    fps.stop()
    print("[INFO] Final FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()