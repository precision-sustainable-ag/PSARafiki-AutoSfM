import pandas as pd

import matplotlib.pyplot as plt
import datetime
import json

# Path to your JSON file
file_path = 'data.json'

# Reading the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)  # This parses the JSON into a Python dictionary/list

# Accessing the metadata
status = data['status']
info = data['info']
collections = data['collections']

# Convert data into a DataFrame to analyze the number of images per camera, per date
image_data = []
locations = []

for image in data["images"]:
    timestamp = datetime.datetime.fromisoformat(image["timestamp"])
    date = timestamp.date()
    camera = image["camera"]
    image_type = image["image_type"]
    locations.append(image["location"])


    image_data.append({"date": date, "camera": camera, "image_type": image_type})

# Converting to DataFrame for easy plotting
location_df = pd.DataFrame(locations, columns=['latitude', 'longitude'])

# Convert to DataFrame
df = pd.DataFrame(image_data)

# Group by date and camera to count the number of images
image_count_per_camera_per_date = df.groupby(['date', 'camera', 'image_type']).size().reset_index(name='image_count')

print(image_count_per_camera_per_date)

# Plotting the locations
plt.figure(figsize=(8, 6))
plt.scatter(location_df['longitude'], location_df['latitude'], color='red', marker='x')

# Label the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Locations of Images Captured')
plt.grid(True)
plt.savefig("location.jpg")