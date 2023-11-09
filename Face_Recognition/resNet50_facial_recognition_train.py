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
from keras.layers import Flatten, Dense

# Set the environment variable to allow duplicated libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load pre-trained VGGFace models (vgg16, resnet50, senet50)
vggface = VGGFace(model='vgg16')
vggface_resnet = VGGFace('resnet50')
vggface_senet = VGGFace(model='senet50')

# Print model summary and information about inputs and outputs
print(vggface.summary())
print('Inputs: ', vggface.inputs)
print('Outputs: ', vggface.outputs)

# Load and process an example image (scalia_photo)
scalia_photo = mpl.image.imread('public_images/happy-young-man-carrying-a-cardboard-box-isolated-picture-id174416612-3737689529.jpg')
scalia_photo.shape

# Initialize MTCNN for face detection
face_detection = mtcnn.MTCNN()
face_roi = face_detection.detect_faces(scalia_photo)
face_roi
x1, y1, width, height = face_roi[0]['box']
x2, y2 = x1 + width, y1 + height
face = scalia_photo[y1:y2, x1:x2]
print(face.shape)

# Create a training dataset using keras.utils.image_dataset_from_directory
train_dataset = keras.utils.image_dataset_from_directory('pubfig/train', shuffle=True, batch_size=8, image_size=(224,224))

# Define data augmentation operations to be applied to the training dataset
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomContrast(0.2),
    keras.layers.RandomBrightness(0.2),
])

# Load a pre-trained VGGFace model (resnet50) and modify it for transfer learning
vggface_resnet_base = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

nb_class = 4  # Number of new people + 1 for unknown/Invalid

# Freeze the base model to prevent retraining its layers
vggface_resnet_base.trainable = False
last_layer = vggface_resnet_base.get_layer('avg_pool').output

# Build a new model for classification
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = vggface_resnet_base(x)
x = Flatten(name='flatten')(x)
out = Dense(nb_class, name='classifier')(x)
custom_vgg_model = keras.Model(inputs, out)
custom_vgg_model.summary()

# Set a learning rate and compile the custom model
base_learning_rate = 0.0001
custom_vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

# Train the custom model on the training dataset
history = custom_vgg_model.fit(train_dataset, epochs=20)

# Save the trained model to a file
custom_vgg_model.save("test4.h5")
