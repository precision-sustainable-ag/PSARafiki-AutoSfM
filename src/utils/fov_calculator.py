import math
import pandas as pd

def calculate_coc(sensor_diagonal: float) -> float:
    """
    Calculate the circle of confusion based on the sensor diagonal.

    Parameters:
    - sensor_diagonal: The diagonal of the camera sensor in mm.

    Returns:
    - CoC: The circle of confusion in mm.
    """
    return sensor_diagonal / 1500

def calculate_hyperfocal(f: float, N: float, CoC: float) -> float:
    """
    Calculate the hyperfocal distance.

    Parameters:
    - f: Focal length in mm.
    - N: Aperture (f-number).
    - CoC: Circle of confusion in mm.

    Returns:
    - H: The hyperfocal distance in mm.
    """
    return (f**2) / (N * CoC) + f

def calculate_depth_of_field(f: float, N: float, D: float, sensor_diagonal: float) -> tuple:
    """
    Calculate the depth of field (near, far, and total).

    Parameters:
    - f: Focal length in mm.
    - N: Aperture (f-number).
    - D: Subject distance in mm.
    - sensor_diagonal: The diagonal of the camera sensor in mm.

    Returns:
    - Dn: Near focus distance in mm.
    - Df: Far focus distance in mm.
    - DOF: Total depth of field in mm.
    """
    CoC = calculate_coc(sensor_diagonal)
    H = calculate_hyperfocal(f, N, CoC)

    Dn = (H * D) / (H + (D - f))
    Df = (H * D) / (H - (D - f)) if (H - (D - f)) != 0 else float('inf')
    DOF = Df - Dn if Df != float('inf') else float('inf')

    return Dn, Df, DOF

def calculate_field_of_view(f: float, sensor_width: float, sensor_height: float, D: float) -> tuple:
    """
    Calculate the field of view in horizontal and vertical dimensions.

    Parameters:
    - f: Focal length in mm.
    - sensor_width: Width of the camera sensor in mm.
    - sensor_height: Height of the camera sensor in mm.
    - D: Subject distance in mm.

    Returns:
    - FOV_h: Horizontal field of view in mm.
    - FOV_v: Vertical field of view in mm.
    """
    theta_h = 2 * math.atan(sensor_width / (2 * f))
    theta_v = 2 * math.atan(sensor_height / (2 * f))
    
    FOV_h = 2 * D * math.tan(theta_h / 2)
    FOV_v = 2 * D * math.tan(theta_v / 2)
    
    return FOV_h, FOV_v



# Example usage
focal_length = 3.04
aperture = 2
sensor_diagonal = 4.6
sensor_width = 3.68  # Width of the camera sensor in mm
sensor_height = 2.76  # Height of the camera sensor in mm

# Prepare data for various subject distances
data = []

# Iterate over subject distances from 0.1m to 10m, incrementing by 0.1m
for D in range(1, 101):  # D in decimeters, i.e., 0.1 to 10 meters
    subject_distance = D * 100  # Convert to mm
    Dn, Df, DOF = calculate_depth_of_field(focal_length, aperture, subject_distance, sensor_diagonal)
    FOV_h, FOV_v = calculate_field_of_view(focal_length, sensor_width, sensor_height, subject_distance)
    data.append({
        "Subject Distance (m)": D / 10,
        "Near Focus Distance (mm)": Dn,
        "Far Focus Distance (mm)": Df if Df != float('inf') else 'Infinity',
        "Depth of Field (mm)": DOF,
        "Horizontal FOV (mm)": FOV_h,
        "Vertical FOV (mm)": FOV_v
    })

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file_path = 'depth_of_field_fov.csv'
df.to_csv(csv_file_path, index=False)