import math
from typing import List
import Metashape

def argsort(arr: List[float]) -> List[int]:
    """
    Returns the indices that would sort an array.

    Args:
        arr (List[float]): Array to sort.

    Returns:
        List[int]: Indices that would sort the array.
    """
    indices = sorted(range(len(arr)), key=lambda idx: arr[idx])
    return indices

class CameraStats:
    """
    Class to calculate and store statistics for a camera in a Metashape chunk.
    """

    def __init__(self, camera: Metashape.Camera) -> None:
        """
        Initializes CameraStats with a given camera and calculates various parameters.

        Args:
            camera (Metashape.Camera): The camera object.
        """
        chunk = camera.chunk

        self.camera = camera
        self.estimated_location = None
        self.estimated_rotation = None
        self.reference_location = None
        self.reference_rotation = None
        self.error_location = None
        self.error_rotation = None
        self.sigma_location = None
        self.sigma_rotation = None

        # If the camera is not transformed, exit
        if not camera.transform:
            return

        # Obtain transformation matrix and coordinate reference system (CRS)
        transform = chunk.transform.matrix
        crs = chunk.crs

        # Transform coordinates to the camera's CRS if necessary
        if chunk.camera_crs:
            transform = Metashape.CoordinateSystem.datumTransform(crs, chunk.camera_crs) * transform
            crs = chunk.camera_crs

        # Get Earth-Centered, Earth-Fixed (ECEF) CRS
        ecef_crs = self.getCartesianCrs(crs)

        # Calculate camera transformation in ECEF coordinates
        camera_transform = transform * camera.transform
        antenna_transform = self.getAntennaTransform(camera.sensor)
        location_ecef = camera_transform.translation() + camera_transform.rotation() * antenna_transform.translation()
        rotation_ecef = camera_transform.rotation() * antenna_transform.rotation()

        # Convert estimated location to the project CRS
        self.estimated_location = Metashape.CoordinateSystem.transform(location_ecef, ecef_crs, crs)
        
        # Calculate location error if reference location is available
        if camera.reference.location:
            self.reference_location = camera.reference.location
            self.error_location = Metashape.CoordinateSystem.transform(self.estimated_location, crs, ecef_crs) - Metashape.CoordinateSystem.transform(self.reference_location, crs, ecef_crs)
            self.error_location = crs.localframe(location_ecef).rotation() * self.error_location

        # Determine the local frame for rotation calculations
        if chunk.euler_angles == Metashape.EulerAnglesOPK or chunk.euler_angles == Metashape.EulerAnglesPOK:
            localframe = crs.localframe(location_ecef)
        else:
            localframe = ecef_crs.localframe(location_ecef)

        # Calculate estimated rotation in local frame
        self.estimated_rotation = Metashape.utils.mat2euler(localframe.rotation() * rotation_ecef, chunk.euler_angles)
        
        # Calculate rotation error if reference rotation is available
        if camera.reference.rotation:
            self.reference_rotation = camera.reference.rotation
            self.error_rotation = self.estimated_rotation - self.reference_rotation
            self.error_rotation.x = (self.error_rotation.x + 180) % 360 - 180
            self.error_rotation.y = (self.error_rotation.y + 180) % 360 - 180
            self.error_rotation.z = (self.error_rotation.z + 180) % 360 - 180

        # Calculate location covariance if available
        if camera.location_covariance:
            T = crs.localframe(location_ecef) * transform
            R = T.rotation() * T.scale()

            cov = R * camera.location_covariance * R.t()
            self.sigma_location = Metashape.Vector([math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), math.sqrt(cov[2, 2])])

        # Calculate rotation covariance if available
        if camera.rotation_covariance:
            T = localframe * camera_transform  # to reflect rotation angles ypr (ecef_crs.localfram) or opk (crs.localframe)
            R0 = T.rotation()

            dR = antenna_transform.rotation()

            da = Metashape.utils.dmat2euler(R0 * dR, R0 * self.makeRotationDx(0) * dR, chunk.euler_angles)
            db = Metashape.utils.dmat2euler(R0 * dR, R0 * self.makeRotationDy(0) * dR, chunk.euler_angles)
            dc = Metashape.utils.dmat2euler(R0 * dR, R0 * self.makeRotationDz(0) * dR, chunk.euler_angles)

            R = Metashape.Matrix([da, db, dc]).t()

            cov = R * camera.rotation_covariance * R.t()

            self.sigma_rotation = Metashape.Vector([math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), math.sqrt(cov[2, 2])])

    def getCartesianCrs(self, crs: Metashape.CoordinateSystem) -> Metashape.CoordinateSystem:
        """
        Returns a Cartesian CRS for the given CRS.

        Args:
            crs (Metashape.CoordinateSystem): Coordinate reference system.

        Returns:
            Metashape.CoordinateSystem: Cartesian CRS.
        """
        ecef_crs = crs.geoccs
        if ecef_crs is None:
            ecef_crs = Metashape.CoordinateSystem('LOCAL')
        return ecef_crs

    def getAntennaTransform(self, sensor: Metashape.Sensor) -> Metashape.Matrix:
        """
        Returns the antenna transform for a given sensor.

        Args:
            sensor (Metashape.Sensor): The sensor object.

        Returns:
            Metashape.Matrix: Antenna transformation matrix.
        """
        location = sensor.antenna.location
        if location is None:
            location = sensor.antenna.location_ref
        rotation = sensor.antenna.rotation
        if rotation is None:
            rotation = sensor.antenna.rotation_ref
        return Metashape.Matrix.Diag((1, -1, -1, 1)) * Metashape.Matrix.Translation(location) * Metashape.Matrix.Rotation(Metashape.Utils.ypr2mat(rotation))

    def makeRotationDx(self, alpha: float) -> Metashape.Matrix:
        """
        Creates a rotation matrix for a rotation around the X-axis.

        Args:
            alpha (float): Rotation angle in radians.

        Returns:
            Metashape.Matrix: Rotation matrix for X-axis.
        """
        sina = math.sin(alpha)
        cosa = math.cos(alpha)
        return Metashape.Matrix([[0, 0, 0], [0, -sina, -cosa], [0, cosa, -sina]])

    def makeRotationDy(self, alpha: float) -> Metashape.Matrix:
        """
        Creates a rotation matrix for a rotation around the Y-axis.

        Args:
            alpha (float): Rotation angle in radians.

        Returns:
            Metashape.Matrix: Rotation matrix for Y-axis.
        """
        sina = math.sin(alpha)
        cosa = math.cos(alpha)
        return Metashape.Matrix([[-sina, 0, cosa], [0, 0, 0], [-cosa, 0, -sina]])

    def makeRotationDz(self, alpha: float) -> Metashape.Matrix:
        """
        Creates a rotation matrix for a rotation around the Z-axis.

        Args:
            alpha (float): Rotation angle in radians.

        Returns:
            Metashape.Matrix: Rotation matrix for Z-axis.
        """
        sina = math.sin(alpha)
        cosa = math.cos(alpha)
        return Metashape.Matrix([[-sina, -cosa, 0], [cosa, -sina, 0], [0, 0, 0]])

    def getEulerAnglesName(self, euler_angles: Metashape.EulerAngles) -> str:
        """
        Returns the name of the Euler angles type.

        Args:
            euler_angles (Metashape.EulerAngles): Euler angles type.

        Returns:
            str: Name of the Euler angles type.
        """
        if euler_angles == Metashape.EulerAnglesOPK:
            return "OPK"
        if euler_angles == Metashape.EulerAnglesPOK:
            return "POK"
        if euler_angles == Metashape.EulerAnglesYPR:
            return "YPR"
        if euler_angles == Metashape.EulerAnglesANK:
            return "ANK"
        return "Unknown"

    def printVector(self, f, name: str, value: Metashape.Vector, precision: int) -> None:
        """
        Writes a vector to a file with a given name and precision.

        Args:
            f: File object to write to.
            name (str): Name of the vector.
            value (Metashape.Vector): Vector values.
            precision (int): Precision for formatting.
        """
        fmt = "{:." + str(precision) + "f}"
        fmt = "    " + name + ": " + fmt + " " + fmt + " " + fmt + "\n"
        f.write(fmt.format(value.x, value.y, value.z))

    def write(self, f) -> None:
        """
        Writes camera statistics to a file.

        Args:
            f: File object to write to.
        """
        euler_name = self.getEulerAnglesName(self.camera.chunk.euler_angles)

        f.write(self.camera.label + "\n")
        if self.reference_location:
            self.printVector(f, "   XYZ source", self.reference_location, 6)
        if self.error_location:
            self.printVector(f, "   XYZ error", self.error_location, 6)
        if self.estimated_location:
            self.printVector(f, "   XYZ estimated", self.estimated_location, 6)
        if self.sigma_location:
            self.printVector(f, "   XYZ sigma", self.sigma_location, 6)
        if self.reference_rotation:
            self.printVector(f, "   " + euler_name + " source", self.reference_rotation, 3)
        if self.error_rotation:
            self.printVector(f, "   " + euler_name + " error", self.error_rotation, 3)
        if self.estimated_rotation:
            self.printVector(f, "   " + euler_name + " estimated", self.estimated_rotation, 3)
        if self.sigma_rotation:
            self.printVector(f, "   " + euler_name + " sigma", self.sigma_rotation, 3)

    def to_dict(self) -> dict:
        """
        Converts camera statistics to a dictionary format.

        Returns:
            dict: Dictionary containing camera statistics.
        """
        row = dict()
        row["label"] = self.camera.label
        
        # Estimated location
        if self.estimated_location is not None:
            row["Estimated_X"] = self.estimated_location[0]
            row["Estimated_Y"] = self.estimated_location[1]
            row["Estimated_Z"] = self.estimated_location[2]

        # Estimated Rotation
        if self.estimated_rotation is not None:
            row["Estimated_Yaw"] = self.estimated_rotation[0]
            row["Estimated_Pitch"] = self.estimated_rotation[1]
            row["Estimated_Roll"] = self.estimated_rotation[2]

        return row

class MarkerStats:
    """
    Class to calculate and store statistics for a marker in a Metashape chunk.
    """

    def __init__(self, marker: Metashape.Marker) -> None:
        """
        Initializes MarkerStats with a given marker and calculates various parameters.

        Args:
            marker (Metashape.Marker): The marker object.
        """
        chunk = marker.chunk

        self.marker = marker
        self.estimated_location = None
        self.reference_location = None
        self.error_location = None
        self.sigma_location = None

        # If the marker position is not set, exit
        if not marker.position:
            return

        # Obtain transformation matrix and coordinate reference system (CRS)
        transform = chunk.transform.matrix
        crs = chunk.crs

        # Transform coordinates to the marker's CRS if necessary
        if chunk.marker_crs:
            transform = Metashape.CoordinateSystem.datumTransform(crs, chunk.marker_crs) * transform
            crs = chunk.marker_crs

        # Get Earth-Centered, Earth-Fixed (ECEF) CRS
        ecef_crs = self.getCartesianCrs(crs)

        # Calculate marker position in ECEF coordinates
        location_ecef = transform.mulp(marker.position)

        # Convert estimated location to the project CRS
        self.estimated_location = Metashape.CoordinateSystem.transform(location_ecef, ecef_crs, crs)
        
        # Calculate location error if reference location is available
        if marker.reference.location:
            self.reference_location = marker.reference.location
            self.error_location = Metashape.CoordinateSystem.transform(self.estimated_location, crs, ecef_crs) - Metashape.CoordinateSystem.transform(self.reference_location, crs, ecef_crs)
            self.error_location = crs.localframe(location_ecef).rotation() * self.error_location

        # Calculate location covariance if available
        if marker.position_covariance:
            T = crs.localframe(location_ecef) * transform
            R = T.rotation() * T.scale()

            cov = R * marker.position_covariance * R.t()
            self.sigma_location = Metashape.Vector([math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), math.sqrt(cov[2, 2])])

    def getCartesianCrs(self, crs: Metashape.CoordinateSystem) -> Metashape.CoordinateSystem:
        """
        Returns a Cartesian CRS for the given CRS.

        Args:
            crs (Metashape.CoordinateSystem): Coordinate reference system.

        Returns:
            Metashape.CoordinateSystem: Cartesian CRS.
        """
        ecef_crs = crs.geoccs
        if ecef_crs is None:
            ecef_crs = Metashape.CoordinateSystem('LOCAL')
        return ecef_crs

    def printVector(self, f, name: str, value: Metashape.Vector, precision: int) -> None:
        """
        Writes a vector to a file with a given name and precision.

        Args:
            f: File object to write to.
            name (str): Name of the vector.
            value (Metashape.Vector): Vector values.
            precision (int): Precision for formatting.
        """
        fmt = "{:." + str(precision) + "f}"
        fmt = "    " + name + ": " + fmt + " " + fmt + " " + fmt + "\n"
        f.write(fmt.format(value.x, value.y, value.z))

    def write(self, f) -> None:
        """
        Writes marker statistics to a file.

        Args:
            f: File object to write to.
        """
        f.write(self.marker.label + "\n")
        if self.reference_location:
            self.printVector(f, "   XYZ source", self.reference_location, 6)
        if self.error_location:
            self.printVector(f, "   XYZ error", self.error_location, 6)
        if self.estimated_location:
            self.printVector(f, "   XYZ estimated", self.estimated_location, 6)
        if self.sigma_location:
            self.printVector(f, "   XYZ sigma", self.sigma_location, 6)

    def to_dict(self) -> dict:
        """
        Converts marker statistics to a dictionary format.

        Returns:
            dict: Dictionary containing marker statistics.
        """
        row = dict()
        row["label"] = self.marker.label
        if self.reference_location:
            row["Reference_X"] = self.reference_location[0]
            row["Reference_Y"] = self.reference_location[1]
            row["Reference_Z"] = self.reference_location[2]
        if self.error_location:
            row["Error_X"] = self.error_location[0]
            row["Error_Y"] = self.error_location[1]
            row["Error_Z"] = self.error_location[2]
        if self.estimated_location:
            row["Estimated_X"] = self.estimated_location[0]
            row["Estimated_Y"] = self.estimated_location[1]
            row["Estimated_Z"] = self.estimated_location[2]
        if self.sigma_location:
            row["Variance_X"] = self.sigma_location[0]
            row["Variance_Y"] = self.sigma_location[1]
            row["Variance_Z"] = self.sigma_location[2]

        return row

def find_object_dimension(focal_length: float, half_width: float, camera_height: float) -> float:
    """
    Calculates the half-width of an object in real-world units.

    Args:
        focal_length (float): Focal length of the camera.
        half_width (float): Half-width of the image plane.
        camera_height (float): Height of the camera from the object.

    Returns:
        float: Half-width of the object in real-world units.
    """
    object_half_width = half_width * (camera_height / focal_length)
    return object_half_width


def get_xy_rotation_matrix(angle: float) -> List[List[float]]:
    """
    Gets the rotation matrix for the Yaw.

    Args:
        angle (float): Yaw angle in degrees.

    Returns:
        List[List[float]]: 2x2 rotation matrix for the given yaw angle.
    """
    alpha = angle * (math.pi / 180.0)
    cosa = math.cos(alpha)
    sina = math.sin(alpha)
    R = [[cosa, -sina],
         [sina, cosa]]

    return R

def rotation_transform(coord: List[float], R: List[List[float]]) -> List[float]:
    """
    Applies a rotation transform to a 2D coordinate vector.

    Args:
        coord (List[float]): Coordinate vector to rotate.
        R (List[List[float]]): Rotation matrix.

    Returns:
        List[float]: Transformed coordinates.
    """
    assert len(coord) == 2

    rotated_coord = [
        R[0][0] * coord[0] + R[0][1] * coord[1], 
        R[1][0] * coord[0] + R[1][1] * coord[1]
    ]
    
    return rotated_coord

def add_lists(a: List[float], b: List[float]) -> List[float]:
    """
    Adds two lists element-wise.

    Args:
        a (List[float]): First list.
        b (List[float]): Second list.

    Returns:
        List[float]: Element-wise sum of the two lists.
    """
    assert len(a) == len(b)
    return [_a + _b for _a, _b in zip(a, b)]

def field_of_view(center_coords: List[float], half_width: float, half_height: float, yaw_angle: float) -> List[float]:
    """
    Calculates the field of view corners for a camera.

    Args:
        center_coords (List[float]): Center coordinates of the camera.
        half_width (float): Half-width of the field of view.
        half_height (float): Half-height of the field of view.
        yaw_angle (float): Yaw angle in degrees.

    Returns:
        List[float]: Coordinates of the field of view corners.
    """
    # These are the coordinates with respect to origin changed to center_coords
    # and rotation=0
    top_right = [half_width, half_height]
    bottom_left = [-half_width, -half_height]
    top_left = [-half_width, half_height]
    bottom_right = [half_width, -half_height]

    # Get rotation matrix
    R = get_xy_rotation_matrix(yaw_angle)

    # Rotate the new coordinates
    rotated_top_right = rotation_transform(top_right, R)
    rotated_bottom_left = rotation_transform(bottom_left, R)
    rotated_top_left = rotation_transform(top_left, R)
    rotated_bottom_right = rotation_transform(bottom_right, R)

    # Translate the coordinates back to the original position
    rotated_top_right = add_lists(rotated_top_right, center_coords)
    rotated_bottom_left = add_lists(rotated_bottom_left, center_coords)
    rotated_top_left = add_lists(rotated_top_left, center_coords)
    rotated_bottom_right = add_lists(rotated_bottom_right, center_coords)

    coordinates = [
        rotated_top_right, rotated_bottom_left, 
        rotated_top_left, rotated_bottom_right
    ]

    # The first two will determine the left side coordinates
    mask = argsort([coord[0] for coord in coordinates])

    # Determine which is the top_left and bottom_left
    left_coordinates = coordinates[mask[0]], coordinates[mask[1]]
    top_sort = argsort([coord[1] for coord in left_coordinates])
    bottom_left = left_coordinates[top_sort[0]]
    top_left = left_coordinates[top_sort[1]]
    
    # Determine which is the top_right and bottom_right
    right_coordinates = coordinates[mask[2]], coordinates[mask[3]]
    top_sort = argsort([coord[1] for coord in right_coordinates])
    bottom_right = right_coordinates[top_sort[0]]
    top_right = right_coordinates[top_sort[1]]

    return (
        top_left[0], top_left[1],
        bottom_left[0], bottom_left[1],
        bottom_right[0], bottom_right[1],
        top_right[0], top_right[1]
    )
