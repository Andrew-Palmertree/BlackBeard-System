import cv2
import os
import numpy as np
import random
import shutil
from PIL import Image 

#This file needs to be editted
#This was for splitting data from the generated folder to the mobleNet model classes
#Used for personal use but may be of use to you

print("This file is to split the data into different folders fromt the augmented data")
user_input = input("Please press\n1 for cascade method or \n2 for mobilenet method:\n")
print("\nYou entered: ", user_input, "\n")

if user_input == '1':
    
    # Create a list of image file extensions (you can extend this list)
    image_extensions = ['.jpg', '.png']
    
    # Initialize a count for the number of images
    image_count = 0
    image_count2 = 0
    
    # The base dir. For example, /home/andrew/facial_recognition. Get the dir path from this file.
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
    #         print(first_part)
    
    # Sort the list in alphanumeric order
    first_parts = sorted(first_parts)
    
    
    
    # Print the first parts and their count
    for i, first_part in enumerate(first_parts):
        image_count=0
        image_count2=0
        print(f"Face {i + 1}: {first_part}")
        image_dir_root = os.path.join(BASE_DIR, f"{first_part}")
        # Define the directory where you want to save the cropped images
        cropped_dir = os.path.join(image_dir_root, f"{first_part}_cropped")
        # Define the directory for generated images
        generated_dir = os.path.join(image_dir_root, f"{first_part}_generated")
        
        # List files in the directory
        for filename in os.listdir(cropped_dir):
            # Check if the file has one of the specified image extensions
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Use Pillow to open and verify the image file
                try:
                    with Image.open(os.path.join(cropped_dir, filename)) as img:
                        img.verify()
                        image_count += 1
                except (IOError, SyntaxError):
                    # The file is not a valid image
                    pass
                
        # List files in the directory
        for filename in os.listdir(generated_dir):
            # Check if the file has one of the specified image extensions
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Use Pillow to open and verify the image file
                try:
                    with Image.open(os.path.join(generated_dir, filename)) as img:
                        img.verify()
                        image_count2 += 1
                except (IOError, SyntaxError):
                    # The file is not a valid image
                    pass
        print(f"Amount in the {first_part}_cropped: ", image_count)
        print(f"Amount in the {first_part}_generated: ", image_count2, "\n")
    
    user_input2 = input("Please enter the amount of images you want to have in the cropped dir (100-250):\n")
    print("\nYou entered: ", user_input2, "\n")
    
    try:
        user_input2 = int(user_input2)
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        exit()

    
    # Create a list of image file extensions (you can extend this list)
    image_extensions = ['.jpg', '.png']
    
    # Initialize a count for the number of images
    image_count = 0
    
    # The base dir. For example, /home/andrew/facial_recognition. Get the dir path from this file.
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
    #         print(first_part)
    
    # Sort the list in alphanumeric order
    first_parts = sorted(first_parts)
    
    
    
    # Print the first parts and their count
    for i, first_part in enumerate(first_parts):
        image_count=0
        print(f"Face {i + 1}: {first_part}")
        image_dir_root = os.path.join(BASE_DIR, f"{first_part}")
        # Define the directory where you want to save the cropped images
        cropped_dir = os.path.join(image_dir_root, f"{first_part}_cropped")
        # Define the directory for generated images
        generated_dir = os.path.join(image_dir_root, f"{first_part}_generated")
        
        # List files in the directory
        for filename in os.listdir(cropped_dir):
            # Check if the file has one of the specified image extensions
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Use Pillow to open and verify the image file
                try:
                    with Image.open(os.path.join(cropped_dir, filename)) as img:
                        img.verify()
                        image_count += 1
                except (IOError, SyntaxError):
                    # The file is not a valid image
                    pass
        
        amount_to_transfer = user_input2 - image_count
        
        # Convert val_count to a whole number (integer)
        amount_to_transfer = int(amount_to_transfer)
        
        print(f"Amount in the {first_part}_cropped: ", image_count)
        print("Amount to transfer: ",amount_to_transfer ,"\n")
        
        # List image files in the source directory
        image_gen_files = [f for f in os.listdir(generated_dir) if f.lower().endswith(('.jpg', '.png'))]
        
        # Shuffle the list of image files
        random.shuffle(image_gen_files)

        # Select the first amount_to_transfer files (you can change this number)
        selected_files = image_gen_files[:amount_to_transfer]

        # move the selected files to the destination directory
        for file_name in selected_files:
            source_file = os.path.join(generated_dir, file_name)
            destination_file = os.path.join(cropped_dir, file_name)
            shutil.copy(source_file, destination_file)
            #change .copy if you want to copy instead of move
        
        
    
elif user_input == '2':
    # Create a list of image file extensions (you can extend this list)
    image_extensions = ['.jpg', '.png']
    
    # Initialize a count for the number of images
    image_count = 0
    
    #val amount images
    val_count = 0
    
    #test amount count
    test_amount = 0
    
    # The base dir. For example, /home/andrew/facial_recognition. Get the dir path from this file.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    #MobileNet Folders
    #Test
    test_dir = os.path.join(BASE_DIR, "Facial Recognition/Facial/test")
    #Val
    val_dir = os.path.join(BASE_DIR, "Facial Recognition/Facial/val")
    

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
        cropped_subdirectories = [subdir for subdir in subdirectories_in_directory if subdir.endswith("_generated")]
        
        if cropped_subdirectories:
            subdirectory_contents.extend(cropped_subdirectories)
            
    # Filter directories that end with "_cropped" and save the first part in the array
    first_parts = []
    for directory in subdirectory_contents:
        if directory.endswith("_generated"):
            first_part = directory.rsplit("_generated", 1)[0]
            first_parts.append(first_part)
    #         print(first_part)
    
    # Sort the list in alphanumeric order
    first_parts = sorted(first_parts)
    
    
    
    # Print the first parts and their count
    for i, first_part in enumerate(first_parts):
        print(f"Face {i + 1}: {first_part}")
        image_dir_root = os.path.join(BASE_DIR, f"{first_part}")
        # Define the directory where you want to save the cropped images
        cropped_dir = os.path.join(image_dir_root, f"{first_part}_cropped")
        # Define the directory for generated images
        generated_dir = os.path.join(image_dir_root, f"{first_part}_generated")
        
        # List files in the directory
        for filename in os.listdir(generated_dir):
            # Check if the file has one of the specified image extensions
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Use Pillow to open and verify the image file
                try:
                    with Image.open(os.path.join(generated_dir, filename)) as img:
                        img.verify()
                        image_count += 1
                except (IOError, SyntaxError):
                    # The file is not a valid image
                    pass
        
        print("The total image count is: ", image_count)
        
        val_count = image_count * 0.15
        
        # Convert val_count to a whole number (integer)
        val_count = int(val_count)
        
        test_count = int(image_count-val_count)
        
        print("The amount for val is: ", val_count)
        print("The amount for test is: ", test_count, "\n")
        
        #MobileNet Folders
        #Test
        test_dir = os.path.join(BASE_DIR, f"Facial Recognition/Facial/test/class{i}")
        #Val
        val_dir = os.path.join(BASE_DIR, f"Facial Recognition/Facial/val/class{i}")
        
        # List image files in the source directory
        image_files = [f for f in os.listdir(generated_dir) if f.lower().endswith(('.jpg', '.png'))]
        
        # Shuffle the list of image files
        random.shuffle(image_files)
        
        # Select the first 15% files (you can change this number)
        selected_files = image_files[:val_count]

        # move the selected files to the destination directory
        for file_name in selected_files:
            source_file = os.path.join(generated_dir, file_name)
            destination_file = os.path.join(val_dir, file_name)
            shutil.move(source_file, destination_file)
            #change .copy if you want to copy instead of move
        
        # List image files in the source directory
        image_files = [f for f in os.listdir(generated_dir) if f.lower().endswith(('.jpg', '.png'))]
        # Filter out files that are in selected_files
        image_files = [file for file in image_files if file not in selected_files]
        
        # Move the image files to the destination directory
        for file_name in image_files:
            source_file = os.path.join(generated_dir, file_name)
            destination_file = os.path.join(test_dir, file_name)
            shutil.move(source_file, destination_file)

else:
    print("Invalid choice. Please enter 1 or 2 for the method.")        
        
        
    
