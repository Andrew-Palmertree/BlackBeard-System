from flask import Flask, request
import os
import requests
import subprocess

app = Flask(__name__)

#Sets the directory for the images to go to(Change to match your directory)
IMAGE_FOLDER = '/home/.../pubfig/train'
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER


@app.route('/upload', methods=['POST'])
def upload_image():
  #Receives a URL from the mobile app in the POST body and converts it to proper characters
  photo_url = request.data.decode('utf-8')

  #Receives the query parameter in the POST and sets that to be the users name
  file_name = request.args.get('FileName')

  #If both are received the process will begin
  if photo_url and file_name:
    #Moves to folder in the path set above named after the users set File
    folder_path = os.path.join(app.config['IMAGE_FOLDER'], file_name)
    #If this folder does not exist this will creates it 
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

      #Fetchs the image based on the uploaded URL
      response = requests.get(photo_url)
                
      #If this process is succesful:
      if response.status_code == 200:
        #We now save the image as "FileName(n).png" in the folder
        #where file name is the user and n is the number of the photo uploaded
        image_number = len(os.listdir(folder_path))
        image_filename = f"{file_name}({image_number}).png"
        image_path = os.path.join(folder_path, image_filename)

        with open(image_path, 'wb') as f:
          f.write(response.content)
          
        #The following lines will call the files required for our training model to 
        #properly register the user to be recognized in our real time face detection thread
        #(I figure we dont need to run the tensorflow_lite.py here and have that saved for the main code)
        subprocess.run(["python", "Crop_images.py"])
        subprocess.run(["python", "new_user.py"])
        subprocess.run(["python", "train.py"])
        subprocess.run(["python", "converter.py"])

          return "Image saved successfully", 200
      else:
          return "Failed to fetch the image", 400
  else:
    return "Invalid request data", 400

#Sets the server for the code(Set IP to be that of your pi)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5353)
