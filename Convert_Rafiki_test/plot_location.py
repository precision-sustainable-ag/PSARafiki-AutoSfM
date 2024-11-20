import json
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import pandas as pd
import os

class ImageLocationPlotter:
    def __init__(self, json_directory, plot_boundaries_file, collection_name):
        self.json_directory = json_directory
        self.plot_boundaries_file = plot_boundaries_file
        self.collection_name = collection_name
        self.camera_locations = None
        self.plot_points = None
        self.plot_geometries = []
        self.aruco_first_instances = []  # Stores first instance of ArUco markers
        
    def load_plot_boundaries(self):
        """Load plot boundary points from JSON file."""
        with open(self.plot_boundaries_file, 'r') as file:
            self.plot_points = json.load(file)
    
    def extract_coordinates_from_json(self, file_path):
        """Extract coordinates and associated info from a given JSON file."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        coordinates = [(image["location"], data["info"], image["aruco_detected"]) for image in data["images"]]
        return coordinates
    
    def collect_camera_locations(self):
        """Collect and sort coordinates and ArUco markers from all JSON files."""
        all_coordinates = []

        for filename in os.listdir(self.json_directory):
            if filename.endswith(".json"):
                file_path = os.path.join(self.json_directory, filename)
                coordinates = self.extract_coordinates_from_json(file_path)
                all_coordinates.extend(coordinates)
        
        # Sort coordinates by the "info" field
        all_coordinates.sort(key=lambda x: x[1])  # Sorting by `data["info"]`
        
        # Prepare DataFrame and identify first instances of ArUco markers
        aruco_seen = set()  # Set to track seen ArUco markers
        locations = []
        for location, info, aruco_detected in all_coordinates:
            data = [location[0], location[1], aruco_detected]
            locations.append(data)
            
            # If the ArUco marker hasn't been seen before, store its first instance
            if aruco_detected not in aruco_seen and aruco_detected != -1:
                self.aruco_first_instances.append((location, aruco_detected))
                aruco_seen.add(aruco_detected)
        
        self.camera_locations = pd.DataFrame(locations, columns=['latitude', 'longitude', 'aruco_detected'])
        
    
    def create_camera_geodataframe(self):
        """Convert the camera locations DataFrame into a GeoDataFrame."""
        geometry = [Point(xy) for xy in zip(self.camera_locations['longitude'], self.camera_locations['latitude'])]
        return gpd.GeoDataFrame(self.camera_locations, geometry=geometry, crs="EPSG:4326")
    
    def create_plot_geometries(self):
        """Create GeoDataFrame from the plot boundary points."""
        for plot_name, corners in self.plot_points.items():
            polygon_points = [
                (corners["corner1"]["longitude"], corners["corner1"]["latitude"]),
                (corners["corner2"]["longitude"], corners["corner2"]["latitude"]),
                (corners["corner3"]["longitude"], corners["corner3"]["latitude"]),
                (corners["corner4"]["longitude"], corners["corner4"]["latitude"]),
                (corners["corner1"]["longitude"], corners["corner1"]["latitude"])  # Closing the polygon
            ]
            polygon = Polygon(polygon_points)
            self.plot_geometries.append(polygon)
        
        return gpd.GeoDataFrame(geometry=self.plot_geometries, crs="EPSG:4326")
    
    def plot_locations_and_boundaries_new(self):
        """Plot the camera locations, ArUco detections, and plot boundaries on a map."""
        # Load plot boundaries
        self.load_plot_boundaries()
        
        # Collect camera locations and ArUco markers
        self.collect_camera_locations()
        
        # Create GeoDataFrames
        gdf = self.create_camera_geodataframe()
        
        # Plotting with dynamic extent based on the combined data
        fig, ax = plt.subplots(figsize=(10, 8))
        arucos = []
        # Plot the camera locations with ArUco detections
        for idx, row in self.camera_locations.iterrows():
            lat, lon = row['latitude'], row['longitude']
            aruco_detected = row.get('aruco_detected', None)
            
            # Plot the camera location
            ax.plot(lon, lat, 'bo', markersize=5)
            
            # If ArUco detected, label the point with its ID
            print(aruco_detected)
            if aruco_detected and aruco_detected not in arucos:
                arucos.append(aruco_detected)
                ax.text(lon, lat, str(aruco_detected), fontsize=9, ha='center', color='green', weight='bold')
        
        # Plot the plot boundaries if needed (this section remains commented)
        # plot_gdf.boundary.plot(ax=ax, color='red', linewidth=2, label="Plot Boundaries")
        
        # Add context tiles for a basemap (optional)
        # ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs=gdf.crs.to_string())
        
        # Set axis labels and title
        ax.set_title(f"Locations of Images with ArUco Markers")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Save and display the plot
        plt.savefig(f"results/{self.collection_name}_location_map_with_aruco.jpg")
        plt.show()

    def plot_locations_and_boundaries(self):
        """Plot the camera locations, ArUco first instances, and plot boundaries on a map."""
        # Load plot boundaries
        self.load_plot_boundaries()
        
        # Collect camera locations and ArUco markers
        self.collect_camera_locations()
        
        # Create GeoDataFrames
        gdf = self.create_camera_geodataframe()
        plot_gdf = self.create_plot_geometries()
        plot_gdf = plot_gdf.to_crs(gdf.crs)  # Ensure the same CRS
        
        # Plotting with dynamic extent based on the combined data
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the camera locations
        gdf.plot(ax=ax, marker='o', color='blue', markersize=5, label="Camera Locations")
        
        # Plot the plot boundaries
        plot_gdf.boundary.plot(ax=ax, color='red', linewidth=2, label="Plot Boundaries")
        
        # # Plot a single point for each unique ArUco marker at its first instance
        # for (location, aruco_detected) in self.aruco_first_instances:
        #     ax.plot(location[1], location[0], 'go', markersize=5, label=f"ArUco {aruco_detected}")
        #     ax.text(location[1], location[0], str(aruco_detected), fontsize=9, ha='right', color='green', weight='bold')
        
        # Add context tiles for a basemap (optional)
        # ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs=gdf.crs.to_string())
        
        # Set axis labels and title
        ax.set_title(f"Locations of Images with ArUco Markers and Plot Boundaries")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Add legend
        ax.legend()
        
        # Save and display the plot
        plt.savefig(f"results/{self.collection_name}_location_map_with_aruco.jpg")
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Instantiate the plotter with the necessary directory and file paths
    plotter = ImageLocationPlotter(
        json_directory='metadata', 
        plot_boundaries_file='data/collection_boundary_new.json', 
        collection_name="82_2"
    )
    
    # Create and display the plot
    plotter.plot_locations_and_boundaries()
