import os
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

from main_face_rec import countImages

# Call the countImages function and unpack the returned values
count_trainable_value, folder_name_list, all_folder_name_list = countImages()



## work on training the model even if there is not enough photos in one dir
# # Trainable
# if folder_name_list == all_folder_name_list:
#     trainable = True
# else:
#     trainable = False

# if trainable == False:
#     for element in all_folder_name_list:
#         if element not in folder_name_list:
#             user_input_train = input(f"Do you want to skip {} to train a new model")
    

class_names_list = sorted(folder_name_list)


def makeLabelFile(new_version):
    
    labels = '\n'.join(class_names_list)
    
    # File path
    file_path = f'Models\\labels\\labelmap_{new_version}.txt'
    
    # Write the content to the file
    with open(file_path, 'w') as f:
        f.write(labels)



def main():
    print("IN new_user file: ")
    print(count_trainable_value)
    print(folder_name_list)
    
    # Directory where the models are stored
    models_directory = 'Models/'
    
    # Initialize an empty list for model numbers
    model_numbers = []

    # Loop through all files in the directory
    for f in os.listdir(models_directory):
        # Check if the file is a model file
        if f.startswith('ResNet50_face_recognition_model_'):
            # Extract the model number from the filename
            model_number = int(f.split("_")[-1].split(".h5")[0])
            # Add the model number to the list
            model_numbers.append(model_number)

    # Find the latest version
    if model_numbers:
        latest_version = max(model_numbers)
    else:
        latest_version = 0

    print("latest_version:")
    print(latest_version)
    
    # Increment the model version
    new_version = latest_version + 1
    new_model_filename = os.path.join(models_directory, f'ResNet50_face_recognition_model_{new_version}.h5')

    print("new_version:")
    print(new_version)
    
    # create new label file
    makeLabelFile(new_version)
    
    

    model = load_model(f'{models_directory}ResNet50_face_recognition_model_{latest_version}.h5')

    new_class_dataset = tf.keras.preprocessing.image_dataset_from_directory('pubfig/train', shuffle=True, batch_size=8, image_size=(224,224), class_names=class_names_list)

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
    nb_class = count_trainable_value  # Number of old classes + 1 for the new class
    x = model.layers[-2].output
    out = Dense(nb_class, name='classifier')(x)
    model = tf.keras.Model(model.input, out)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(new_class_dataset, epochs=15)

    model.save(new_model_filename)
    


if __name__ == "__main__":
    main()

