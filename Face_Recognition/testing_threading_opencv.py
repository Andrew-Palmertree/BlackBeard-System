# Import necessary libraries
import cv2
from threading import Thread
from imutils.video import FPS
import time

# Define a class for capturing video from the camera
class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=32):
        # Initialize the camera with specified resolution and framerate
        self.camera = cv2.VideoCapture(1)
        self.camera.set(3, resolution[0])
        self.camera.set(4, resolution[1])
        self.camera.set(5, framerate)
        self.frame = None
        self.stopped = False

    def start(self):
        # Start a new thread to continuously capture frames
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            # Capture frames from the camera
            ret, frame = self.camera.read()
            if not ret:
                break
            self.frame = frame

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Stop the thread capturing frames
        self.stopped = True

def main():
    # Create an instance of PiVideoStream
    vs = PiVideoStream().start()
    time.sleep(2.0)  # Allow the camera to warm up
    fps = FPS().start()  # Initialize the FPS counter

    # Initialize time variables for calculating FPS
    start_time = time.time()
    interval = 5  # Display FPS every 5 seconds

    while True:
        frame = vs.read()  # Read a frame from the camera

        if frame is not None:
            # Display the frame
            cv2.imshow("Frame", frame)

        # Check for user input ('q' to exit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        fps.update()

        # Display FPS every 'interval' seconds
        if (time.time() - start_time) > interval:
            fps.stop()
            print("[INFO] FPS: {:.2f}".format(fps.fps()))
            fps.start()
            start_time = time.time()

    fps.stop()
    print("[INFO] Final FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
