Hello!
Note the .h5 tensorflow models and tenserflow lite models couldn't be uploaded to github. unzip the pubfig folder and run the train.py and then the convert.py files to get the models. 
The name of the tensorrflow model is at the end of tensorflow.py. Copy that name and make sure it matches the model name at the beginning of the convert.py file.
--------------------------------------------------------------------
Make sure you have the following packages installed and the exact version:

very important: make sure to have the 64 bit OS Bullseye 11!!!!

tensorflow==2.10.0

opencv-contrib-python==4.7.0.72

numpy==1.26.2

flatbuffers==2.0

pip install git+https://github.com/rcmalli/keras-vggface.git

pip install keras_apptications

You can follow the Virtual_Environment_Setup.md file for help.

------------------------------------------------------------------
Break down of the files:

All photos need to be uploaded into the same directory named pubfig/train. Each sub-folder in pubfig/train is a different peron's face. 
The names of the sub-folders are 1-name1 2-name2 and so on. That way when a new user is added it won't mess up the order in which the model was trained.

Use:
train.py
to train the tensorflow model with pubfig/train data.

Now you can test your model with:
tensorflow.py

If the model gives accurate results when testing go ahead and convert the tensorflow model to a tensorflow lite model:
converter.py

Now run the tensorflow lite model with:
tensorflow_lite.py
(reason why we want to run it in tensorflow lite beacause this will allow us to run the model with the Google TPU to speed up real-time processing)

to add a new user or train the model with more data but have the same amount of users run:
new_user.py
NOTE:
This line of code updates the amount of users in the system. So with the data in pubfig/train there are only 4 users so change this variable to 4 or the number of users you have.
nb_class = 6  # Number of old classes + 1 for the new class

Change the "class_name" to have the right number of users equal to "nb_class" from above.
new_class_dataset = tf.keras.preprocessing.image_dataset_from_directory('pubfig/train', shuffle=True, batch_size=8, image_size=(224,224), class_names=['1-Andrew', '2-Parbin', '3-Unknown', '4-Will', '5-Hannah', '6-Suzanne'])

--------------------------------------------------------------------
Extra:
We can use the:
take_photos.py
to capture more data for pubfig/train folder
you will need to manually move the photos for now.

Use:
Crop_images.py
to crop the images into 300x300 pixel images with just the face

THAT'S it. :)
