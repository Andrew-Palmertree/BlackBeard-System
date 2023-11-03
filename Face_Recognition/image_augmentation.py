import os
import glob
import cv2
import numpy as np

def rotate(image, angle):
    # Rotate the input image by the specified angle (degrees)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    return rotated

def scale(image, scale_factor):
    # Scale the input image by the specified factor
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

def translate(image, x_translation, y_translation):
    # Translate (shift) the input image by the specified x and y values
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    translated = cv2.warpAffine(image, M, (cols, rows))
    return translated

def random_brightness(image):
    # Adjust the brightness of the input image by a random factor
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.random.uniform(0.75, 1.6)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened   

def random_zoom(image):
    # Apply random zoom to the input image
    zoom_factor = np.random.uniform(0.65, 1.3)
    zoomed = scale(image, zoom_factor)
    return zoomed


def apply_gamma_correction(image, gamma_range=(0.5, 1.5)):
    gamma = np.random.uniform(*gamma_range)
    adjusted_image = np.power(image / 255.0, gamma) * 255.0
    return adjusted_image

def apply_rgb_shift(image, shift_range=(-5, 5)):
    r_shift = np.random.randint(*shift_range)
    g_shift = np.random.randint(*shift_range)
    b_shift = np.random.randint(*shift_range)
    shifted_image = np.clip(image + [b_shift, g_shift, r_shift], 0, 255)
    return shifted_image

def apply_augmentations(image, output_dir, num_samples, image_file):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the input image's base filename (without extension)
    original_image_name, _ = os.path.splitext(os.path.basename(image_file))

    # Get the dimensions of the input image
    rows, cols, _ = image.shape

    for i in range(num_samples):
        augmented_image = image.copy()

        # Apply transformations
        augmented_image = rotate(augmented_image, np.random.uniform(-3, 3)) #For tenesorflow random rotation agnles (1)
        #augmented_image = rotate(augmented_image, np.random.choice([0, 90, 180, 270]))  # Randomly select a flip angle for cascade (2)
    
        augmented_image = random_zoom(augmented_image)
        augmented_image = random_brightness(augmented_image)
        
        #augmented_image = cv2.flip(augmented_image, np.random.randint(0, 2))  # Randomly flip horizontally (1)
        #augmented_image = cv2.flip(augmented_image, np.random.randint(0, 2))  # Randomly flip vertically (1)

        augmented_image = translate(augmented_image, np.random.randint(-15, 15), np.random.randint(-15, 15))
        augmented_image = apply_gamma_correction(augmented_image)
        augmented_image = apply_rgb_shift(augmented_image)

        # Resize the augmented image to the same dimensions as the input image
        augmented_image = cv2.resize(augmented_image, (cols, rows))

        # Save the augmented image with the original image name and a unique index
        filename = os.path.join(output_dir, f"{original_image_name}_{i}.jpg")
        cv2.imwrite(filename, augmented_image)
        
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Get a list of all directories in BASE_DIR and sort them
all_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

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

# Iterate through the first_parts and generate augmented images
for first_part in first_parts:
    image_dir_root = os.path.join(BASE_DIR, first_part)
    cropped_dir = os.path.join(image_dir_root, f"{first_part}_cropped")
    output_directory = os.path.join(image_dir_root, f"{first_part}_generated")
    num_samples = 60  # Number of augmented images to generate

    if os.path.exists(cropped_dir):
        for image_file in glob.glob(os.path.join(cropped_dir, "*.[jp][pn][ge]*")):
            image = cv2.imread(image_file)
            
            if image is not None:
                os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist
                apply_augmentations(image, output_directory, num_samples, image_file)
                print(f"{num_samples} augmented images saved in {output_directory}")

