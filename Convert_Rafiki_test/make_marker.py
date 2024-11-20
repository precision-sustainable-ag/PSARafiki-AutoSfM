import cv2
import numpy as np

def create_aruco_marker(marker_id, dictionary_type=cv2.aruco.DICT_4X4_1000, marker_size=300, save_path=None):
    # Load the pre-defined dictionary
    # aruco_dict = cv2.aruco.Dictionary(dictionary_type)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    # Create the marker image with the specified ID
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Display the marker image
    cv2.imshow(f"ArUco Marker ID {marker_id}", marker_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the marker as an image file if a path is provided
    if save_path:
        cv2.imwrite(save_path, marker_image)
        print(f"ArUco marker ID {marker_id} saved as {save_path}")

# Example usage:
# Create and save a marker with ID 190
create_aruco_marker(17, dictionary_type=cv2.aruco.DICT_4X4_1000, marker_size=300, save_path="aruco_marker_17.png")


# Create and save a marker with ID 130
# create_aruco_marker(130, dictionary_type=cv2.aruco.DICT_4X4_1000, marker_size=300, save_path="aruco_marker_130.png")
