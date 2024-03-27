Hello! :)

This is a break down of the files and what they are used for.
Hopefully this helps!

Quick note:

-for the Unknown class I found the images from this google drive:
https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ

This is the page that I found the google link:
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

-test4.h5
is the tensorflow model I trained for the demo

-vggface_resnet_tflite
is the tensorflow lite model used for the demo


- mmod_human_face_detector.dat 
is for dlib but not used
might use it in the future

- the cascade folder contains the haar cascade face detection algorithm used


important files:

- taking_photos_labels.py:
****** import file *****
this file is used to take photos as the whole image or just the face
when running this press:
s: to capture one whole image (user/user_images)
c: to continuously capute images until you reached the max count givin in 'number_images' (user/user_images)
r: this will continuously capture images of just the face and save the images in (user/user_cropped)

- resNet50_facial_recognition_train.py:
******* This file is used to train the ResNet50 model *********
NOTE: LOOK UNDER Face_Recognition\pubfig\train (<<FOLDER FOR IMAGES)
THERE ARE FOLDER WITH NAMES
each folder represents a class
When I was training the model the folders was organized as:
Andrew (-> class0)
Parbin (-> class1)
Unknown (-> class2)
Will (-> class3)
When I put the folder on github it looked like it shuffled the folder layout
When you train the model keep note of how the order of the files shown in the folder
If the folders are organized differently WHEN YOU TRAIN the model then change the order of the class in the resNet50_tflite_real_time_thread_face_detection.py
you will need to edit to match the folders in Face_Recognition\pubfig\train:
you can also add new folders here to add new users to the model

# Map the class with the highest probability to a name
        if max_prob_class == "Class 0":
            name = "Andrew"
        elif max_prob_class == "Class 1":
            name = "Parbin"
        elif max_prob_class == "Class 2":
            name = "Unknown"
        elif max_prob_class == "Class 3":
            name = "Will"

^^NOTE: WHEN TRAINING I NOTICED THAT HAVING AROUND 30-50 PHOTOS GAVE GOOD RESULTS. MORE OR LESS PHOTOS FOR SOME REASON GAVE BAD RESULTS.^^

- convert_tensorflow_model_to_lite.py:
********* This file will convert the tensorflow model to tensortflow lite for the python to use *******
Saves the lite model to "vggface_senet50_tflite" but feel free to change the name

- resNet50_tflite_real_time_thread_face_detection.py:
******* This is the file where you load the tensorflow lite model for real time use ***********
This file uses threading, opencv, tensorflow, and keras
There is a FPS counter commented out incase you want to see the FPS count when closing the application down


- facial_recognition_resnet_test_real_time.py:
Test the tensorlfow model in real time
note this is not tensorflow lite but regular tensorflow with the .h5 file
this won't work in the pi but will work on the computer


- facial_recognition_resnet_test_on_photo.py:
This file is to take the tensorflow model that was trained and test the results
This won't work in the pi but will work on the computer
There is also some lines of code to test the model against a single image (public_images/2.jpg)






Side files:
These are not important but had them in here just incase

- Crop_images.py:
This file you specify the folder path to where you have your images and will find any faces in the file and save just the face as the original file name
This will delete any files it cannot find a face
doesn't need to be used for the ResNet50 model

- testing_threading_opencv.py:
This file is just to test threading with opencv and see the effects

- split_augmented_data.py:
This file was used more for my personal use to split the generated data from the augmented file

- face_trainning_cascade_recognizer.py:
This will take the images from all the Users/user_cropped dir and train a opencv cascade model
It will save the results to a yaml file
When you have a couple of hundred images the yaml file gets to be huge and still not as accurate as the ResNet50 model
The folder recognizer and pickles are used for this model

- threading_face_detection_cascade.py:
Used to detect faces with the cascade model. Again this model doesn't work as well as the ResNet50 model
The folder recognizer and pickles are used for this model

- image_augmentation.py:
augment the data for the cascade model
not needed for the ResNet model



