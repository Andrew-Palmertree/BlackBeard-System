# Blackbeard-System
This code is for our secure package delivery system, The Blackbeard System. This code will include object and facial recognition along with code for our servo control.



The idea behind Blackbeard is that it will allow delivered packages to end up securely inside of your home versus being left on your porch or inside some clunky metal box. The BlackBeard system incorporates a front door with a package drop located in the lower half of the door, which will drop the package in front of the front door inside of the home. It can fit packages equal to or smaller than 18x20. A camera with the system will be used to detect delivery drivers using artificial intelligence. Upon detection, the chute will open automatically for the delivery driver. The delivery man can put your package inside, then it will sense when no more packages are being delivered and automatically close, dropping the packages inside of the Blackbeard owner’s home. 
This product has space in quite a large market, to be installed on existing homes by DIYers or professional installation by contractors, or with over 1.8 million houses being built a year. This can also be installed on homes being newly built. As a company we will offer basic dimensions and different door materials, or interested customers can request custom dimensions to match their home.
The biggest areas of focus on this project are being able to depict delivery drivers using object recognition and interfacing the servo motors that control the package slip at the bottom of the front door. 
Time permitting, we would also like to implement a type of keyless entry into the home. This method would involve facial recognition of residents (utilizing the same camera for package deliveries) and/or a wireless entry system that unlocks the door when a keyfob is within proximity of the door (similar to keyless entry on modern automobiles). Despite our desire to add this additional feature, the main focus of the project is to get the package delivery system fully functional before pursuing any additional features. 



The goal of our proof of concept is to get the backend code fully functional with the servo motors, camera, & motion sensor interfaced properly with the Raspberry Pi. We plan to use a Raspberry Pi 3b+ for demonstration purposes, and it is likely we will fabricate our own PCB with a microcontroller when it comes time to create our prototype. We plan to implement computer vision with the camera and Raspberry Pi to begin training our system to detect delivery drivers and to recognize different faces. The system will be programmed using either Python or C++ depending on the route we choose to take in terms of an object recognition library. The primary goal is to get the package delivery system functional first, and time permitting we will pursue our secondary goal of creating a functional keyless entry system into the home. Below is a shortened, itemized list of what we want to be able to accomplish in our proof of concept. More detail into the major components is covered later on in this section. It is worth noting that we plan on testing all the components individually.
- Successfully interface components with Raspberry Pi
- Train a neural network to recognize delivery drivers with packages
- Camera is able to transmit real-time images to the Raspberry Pi
- Be able to trigger opening/closing of the package chute upon detection of a delivery driver
- Motion sensor is able to successfully detect motion inside the package chute
- After a period of time with no new motion detected, the chute will close
