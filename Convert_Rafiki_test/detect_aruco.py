import cv2
import numpy as np
print(cv2.__version__)

def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height))

def warp_perspective_to_flatten(image, src_points, dst_points):
    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Warp the perspective
    warped_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    
    return warped_image

def preprocess_image(image, method='none'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'blur':
        return cv2.GaussianBlur(gray, (5, 5), 0)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    elif method == 'threshold':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'dilate':
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(gray, kernel, iterations=1)
    elif method == 'sharpen':
        gaussian_blur = cv2.GaussianBlur(gray, (9, 9), 10.0)
        return cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)
    elif method == "binarize":
        return binarize(gray)
    else:
        return gray
    
def binarize(gray):
    # Apply a slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred)
    _, binary_img = cv2.threshold(contrast_enhanced, 128, 255, cv2.THRESH_BINARY)
    inverted_binary_img = np.where(binary_img == 0, 255, 0).astype(np.uint8)
    return inverted_binary_img


def preprocess_image_for_angles(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    # Apply Canny Edge Detection to emphasize edges
    edges = cv2.Canny(contrast_img, 100, 200)
    return edges

def noise_reduction_aruco_detection(image, dictionary_type=cv2.aruco.DICT_4X4_1000):
    # Apply noise reduction (median blur is good for reducing salt-and-pepper noise)
    blurred_image = cv2.medianBlur(image, 5)
    return blurred_image

def contour_based_aruco_detection(image, dictionary_type=cv2.aruco.DICT_6X6_250):
    aruco_dict = cv2.aruco.Dictionary_get(dictionary_type)
    parameters = cv2.aruco.DetectorParameters_create()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Look for quadrilateral shapes
            x, y, w, h = cv2.boundingRect(approx)
            roi = image[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_roi, aruco_dict, parameters=parameters)
            
            if ids is not None:
                print(f"Detected marker IDs in contour: {ids.flatten()}")
                cv2.aruco.drawDetectedMarkers(roi, corners, ids)
                cv2.imshow("Detected ArUco Marker in Contour", roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return  # Exit after detecting markers
    print("No ArUco markers detected in contours.")

def detect_aruco_markers(image_path, dictionary_type=cv2.aruco.DICT_4X4_1000):
    # Load the image
    image = cv2.imread(image_path)

    preprocessed_img = preprocess_image(image, method='blur')
    # preprocessed_img = image
    # preprocessed_img = contour_based_aruco_detection(image)

    # Load the pre-defined dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    # Create parameters for ArUco marker detection
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect the markers in the image
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(preprocessed_img, aruco_dict, parameters=parameters)
    
    # Check if markers are detected
    if ids is not None:
        # Draw detected markers on the image
        cv2.aruco.drawDetectedMarkers(preprocessed_img, corners, ids)
        
        # Print the detected marker ids
        print("Detected marker IDs:", ids.flatten())
    else:
        print("No ArUco markers detected.")
    
    # Display the resulting image
    cv2.imshow('Aruco Markers Detected', preprocessed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# detect_aruco_markers('/home/mkutuga/Convert_Rafiki_test/data/camera_3_2024-08-14/rgb_camera_3_2024-08-14T10:23:35.745_rgb8.jpg')
# detect_aruco_markers('/home/mkutuga/Convert_Rafiki_test/data/camera_3_2024-08-14/rgb_camera_3_2024-08-14T10:18:14.717_rgb8.jpg')
detect_aruco_markers('/home/mkutuga/Convert_Rafiki_test/IMG_5239.JPG')