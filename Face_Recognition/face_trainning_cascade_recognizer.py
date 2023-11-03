import os
import numpy as np
import re 
# import dlib # (may use this in the future)
import cv2
import pickle
from PIL import Image

# The base dir. For example, /home/andrew/facial_recognition. Get the dir path from this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Get a list of all directories in BASE_DIR and sort them
all_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

# Initialize a count variable to keep track of the number of first_part
count = 0

#create a count to see how many images were recognized and saved to the model
count_image_parbin = 0
count_image_will = 0
count_image_andrew = 0

# List all directories in the specified directory
contents = os.listdir(BASE_DIR)

# Filter the list to include only directories
child_directories = [item for item in contents if os.path.isdir(os.path.join(BASE_DIR, item))]

subdirectory_contents = []

# Iterate through the subdirectories and print their contents
for directory in child_directories:
    directory_path = os.path.join(BASE_DIR, directory)
    directory_contents = os.listdir(directory_path)
    # Filter the list of subdirectories within this subdirectory
    subdirectories_in_directory = [item for item in directory_contents if os.path.isdir(os.path.join(directory_path, item))]
    
    # Check if any subdirectory ends with "_cropped" and append it to subdirectory_contents
    cropped_subdirectories = [subdir for subdir in subdirectories_in_directory if subdir.endswith("_cropped")]
    
    if cropped_subdirectories:
        subdirectory_contents.extend(cropped_subdirectories)

# Filter directories that end with "_cropped" and save the first part in the array
first_parts = []
for directory in subdirectory_contents:
    if directory.endswith("_cropped"):
        first_part = directory.rsplit("_cropped", 1)[0]
        first_parts.append(first_part)

# Sort the list in alphanumeric order
first_parts = sorted(first_parts)

def extract_all_numeric_parts(filename):
    numeric_parts = re.findall(r'\d+', filename)
    if numeric_parts:
        return [int(part) for part in numeric_parts]
    return [0]


#cascade classifier
face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')


# Import LBPHFaceRecognizer from cv2.face (contrib)
from cv2.face import LBPHFaceRecognizer

# Initialize LBPHFaceRecognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Initialize a list to store the labels of the faces detected
count = 0

current_id = 0
label_ids = {}

y_labels = []
x_train = []


# Print the user that's being processed
for i, first_part in enumerate(first_parts):
    print(f"User {i + 1}: {first_part}")
    image_dir_root = os.path.join(BASE_DIR, f"{first_part}")
    # Define the directory where you want to save the cropped images
    cropped_dir = os.path.join(image_dir_root, f"{first_part}_cropped")

    # Reset count to zero for each directory
    count = 0

    # NOTE: this will go through all the photos in the user_input_cropped folder
    for root, dirs, files in os.walk(cropped_dir):
        # Sort the files numerically
        files = sorted(files, key=lambda x: extract_all_numeric_parts(x))
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                
                label = first_part  # Use the user_input as the label 
                #(no need to have a label file just make sure the files are named correctly)

                print(path)
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]

                pil_image = Image.open(path).convert("L")  # Grayscale

                # Resize the image to the desired size
                size = (224, 224)
                final_image = pil_image.resize(size, Image.LANCZOS)

                # Convert the PIL image to a NumPy array
                image_array = np.array(final_image, "uint8")

                # cascade classifier for face detection
                faces = face_classifier.detectMultiScale(image_array, 1.3, 5)

                #removes the image if no face is detected
                if len(faces) == 0:
                    No faces detected, add the image path to the list of images to delete
                        images_to_delete.append(path)

                    try:
                        os.remove(path)
                        print(f"Deleted: {path}")
                    except OSError as e:
                        print(f"Error deleting {path}: {e}")

                else:

                    #Cascade Classifier crops just the face
                    for (x, y, w, h) in faces:
                        cropped_face = image_array[y:y+h, x:x+w]

                        if not cropped_face.size:
                            continue


                        x_train.append(cropped_face)
                        y_labels.append(id_)  # Replace id_ with the appropriate label or identifier
                        print("---------saved--------")  # troubleshooting print statement so you know what images are saved

                        if f"{first_part}" == "Will":
                            count_image_will += 1
                            print("Will: " + str(count_image_will))
                        if f"{first_part}" == "Parbin":
                            count_image_parbin += 1
                            print("Parbin: " + str(count_image_parbin))
                        if f"{first_part}" == "Andrew":
                            count_image_andrew += 1
                            print("Andrew: " + str(count_image_andrew))




                    #Dlib (may use this in the future)
                    # Loop over the detected faces and crop them
#                     for i, face in enumerate(faces):
#                         x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
#                         cropped_face = gray_image_cropped[y:y + h, x:x + w]
# 
#                         if not cropped_face.size:
#                             continue
# 
#                         # Resize each cropped face to a common size, e.g., (100, 100)
#                         cropped_face = cv2.resize(cropped_face, (50, 50))
# 
#                         # Append the cropped face to x_train and add the corresponding label to y_labels
#                         x_train.append(cropped_face)
#                         y_labels.append(id_)  # Replace id_ with the appropriate label or identifier
#                         print("1")

                # Keep count of the images that are saved in for the present user
                count += 1


# Create the directory to store the pickle file if it doesn't exist
pickle_dir = os.path.join(BASE_DIR, "pickles")
if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)

# save the labels to the pickel file    
with open(os.path.join(pickle_dir, "face-labels_Team_test_3.pickle"), 'wb') as f:
    pickle.dump(label_ids, f)
print("\n\n label_ids: " + str(label_ids))

# face recognizer model directory
recognizer_dir = os.path.join(BASE_DIR, "recognizer")
if not os.path.exists(recognizer_dir):
    os.makedirs(recognizer_dir)

# train and save the model in a .yaml file     
recognizer.train(x_train, np.array(y_labels))
recognizer.save(recognizer_dir + "/trainnerTest_Team_test_3.yml")

print("\n\n y_labels: " + str(y_labels)) # print the y_labels for troubleshooting

# print the total count of images that saved for Will, Parbin, and Andrew for troubleshooting purposes
print("Will: " + str(count_image_will))
print("Parbin: " + str(count_image_parbin))
print("Andrew: " + str(count_image_andrew))
