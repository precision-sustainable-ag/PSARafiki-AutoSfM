import os
import cv2
from pathlib import Path

def convert_bgr_to_rgb(image_path, output_dir):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save the image in RGB format to the output path
    output_name = Path(image_path).stem + ".png"
    output_path = Path(output_dir, output_name)
    cv2.imwrite(output_path, rgb_image)

def process_directory(directory, output_dir):
    # Supported image file extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Check if the file is an image
        if filename.lower().endswith(supported_extensions):
            print(f"Processing {filename}...")

            # Convert the image from BGR to RGB
            convert_bgr_to_rgb(file_path, output_dir)

            print(f"Converted {filename} to RGB and saved as {output_dir}")

if __name__ == "__main__":
    # Specify the directory containing the images
    image_directory = "/home/mkutuga/Convert_Rafiki_test/data/pres_images/"
    output_dir = Path("data",Path(image_directory).name)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Process the directory
    process_directory(image_directory, output_dir)
