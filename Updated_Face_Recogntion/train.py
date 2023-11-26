import tensorflow as tf
import keras_vggface
from tensorflow.keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Set the environment variable to allow duplicated libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load a pre-trained VGGFace model (resnet50) and modify it for transfer learning
vggface_resnet_base = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

nb_class = 5  # Number of new people + 1 for unknown/Invalid

# Freeze the base model to prevent retraining its layers
vggface_resnet_base.trainable = False
last_layer = vggface_resnet_base.get_layer('avg_pool').output

# Build a new model for classification
inputs = tf.keras.Input(shape=(224, 224, 3))
x = vggface_resnet_base(inputs)
dropout_rate = 0.5  # Example dropout rate
x = Flatten(name='flatten')(x)
dropout_rate = 0.5  # Example dropout rate
out = Dense(nb_class, name='classifier')(x)
custom_vgg_model = tf.keras.Model(inputs, out)
custom_vgg_model.summary()


# Create a training dataset using keras.utils.image_dataset_from_directory
train_dataset = tf.keras.preprocessing.image_dataset_from_directory('pubfig/train', shuffle=True, batch_size=8, image_size=(224,224), class_names=['1-Andrew', '2-Parbin', '3-Unknown', '4-Will', '5-Hannah'])


# Define a function to apply random brightness adjustment to images
def random_brightness(image):
    # Convert the image to float and apply random brightness
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.random_brightness(image, max_delta=0.2)  # Adjust the max_delta as needed

# Define data augmentation operations to be applied to the training dataset
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
    tf.keras.layers.Lambda(lambda x: random_brightness(x)),  # Apply random brightness
])

# Compile the custom model
base_learning_rate = 0.0001
custom_vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the custom model using the train_generator
# train_dataset_augmented = train_dataset.map(lambda x, y: (data_augmentation(x), y))
history = custom_vgg_model.fit(train_dataset, epochs=20)

# Save the trained model to a file
custom_vgg_model.save("ResNet50_with_Augmentation_pi4.h5")

