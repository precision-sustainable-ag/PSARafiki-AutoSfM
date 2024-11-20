import cv2
import numpy as np

# Load the image
image = cv2.imread('IMG_5239.JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a list of dictionaries with different sizes
aruco_dicts = [
    # cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    # cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
    # cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),
]

# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters()

# Iterate through possible dictionaries
for i, aruco_dict in enumerate(aruco_dicts):
    # Detect the markers in the image
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if ids is not None:
        # If markers are detected, print the dictionary and ID
        # print(f"Detected using DICT_4X4_{50 * (i + 1)}")
        print("Detected marker ID:", ids.flatten())
        
        # Draw the detected markers on the image
        # cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Display the resulting image
        # cv2.imshow('Detected Markers', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # break
else:
    print("No markers detected")