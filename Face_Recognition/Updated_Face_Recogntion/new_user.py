import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

model = load_model('Updated_ResNet50_with_Augmentation_pi3.h5')

new_class_dataset = tf.keras.preprocessing.image_dataset_from_directory('pubfig/train', shuffle=True, batch_size=8, image_size=(224,224), class_names=['1-Andrew', '2-Parbin', '3-Unknown', '4-Will', '5-Hannah', '6-Suzanne'])

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
nb_class = 6  # Number of old classes + 1 for the new class
x = model.layers[-2].output
out = Dense(nb_class, name='classifier')(x)
model = tf.keras.Model(model.input, out)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(new_class_dataset, epochs=10)


model.save("Updated_2_ResNet50_with_Augmentation_pi3.h5")
