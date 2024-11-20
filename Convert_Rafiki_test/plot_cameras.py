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

# Extracting timestamps and cameras
timestamps = []
cameras = []

for image in data["images"]:
    timestamps.append(datetime.datetime.fromisoformat(image["timestamp"]))
    cameras.append(image["camera"])

# Converting camera names to numeric values for plotting
camera_mapping = {camera: idx for idx, camera in enumerate(set(cameras))}
camera_values = [camera_mapping[camera] for camera in cameras]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(timestamps, camera_values, color='blue', marker='o')

# Label the plot
plt.xlabel('Timestamp')
plt.ylabel('Camera')
plt.title('Timestamps by Camera')
plt.yticks(list(camera_mapping.values()), list(camera_mapping.keys()))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("timestamp_by_camera.jpg")
# Display the plot
plt.show()


