from pathlib import Path
import logging
from copy import deepcopy
from typing import Callable, List, Dict, Optional

import Metashape as ms

from .resize import glob_multiple_extensions
from .callbacks import percentage_callback
from .dataframe import DataFrame
from .estimation import CameraStats, MarkerStats, field_of_view, find_object_dimension

log = logging.getLogger(__name__)

class SfM:
    def __init__(self, cfg: Dict) -> None:
        """
        Initializes the SfM (Structure from Motion) class with the given configuration.

        Args:
            cfg (Dict): Configuration dictionary containing project settings.
        """
        self.cfg = cfg
        self.project_name = self.cfg['project_name']
        self.crs = self.get_crs_str()
        self.markerbit = self.get_target_bit()
        self.metashape_key = cfg['metashape_key']
        self.project_path = Path(self.cfg['paths']['agi_project_dir'], f"{self.project_name}.psx")
        self.doc = self.load_or_create_project()
    
        self.num_gpus = (
            cfg['asfm']['num_gpus']
            if cfg['asfm']['num_gpus'] != "all"
            else 2 ** len(ms.app.enumGPUDevices()) - 1
        )

    def get_target_bit(self) -> ms.TargetType:
        """
        Returns the marker bit type for target detection.

        Returns:
            ms.TargetType: Marker bit type.
        """
        return ms.CircularTarget14bit

    def get_crs_str(self) -> str:
        """
        Returns the coordinate reference system (CRS) as a string.

        Returns:
            str: Coordinate reference system string.
        """
        crs = "LOCAL"
        log.info(f"CRS: {crs}")
        return crs

    def load_or_create_project(self) -> ms.Document:
        """
        Opens an existing project or creates and saves a new project if it doesn't exist.

        Returns:
            ms.Document: Metashape Document containing the project.
        """
        if not ms.app.activated:
            ms.License().activate(self.metashape_key)
        
        doc = ms.Document()
        project_name = self.project_name

        if self.project_path.exists():
            log.info(f"Metashape project already exists for {project_name}. Opening project file.")
            doc.open(str(self.project_path), read_only=False, ignore_lock=True)
        else:
            log.info(f"Creating new Metashape project for {project_name}")
            doc.addChunk()
            doc.save(str(self.project_path))

        return doc

    def save_project(self) -> None:
        """
        Saves the current project to the specified path.
        """
        assert self.project_path.suffix == ".psx", "Invalid project file extension."
        self.doc.save()

    def add_photos(self) -> None:
        """
        Adds photos from a directory to the project.
        """
        file_extensions = ['jpg', 'JPG', 'png', 'PNG', 'bmp', 'BMP']
        photos = [str(x) for x in glob_multiple_extensions(self.cfg['paths']['down_photos'], file_extensions)]
        print(photos)

        if self.doc.chunk is None:
            self.doc.addChunk()
        self.doc.chunk.crs = ms.CoordinateSystem(self.crs)
        self.doc.chunk.addPhotos(photos)

    def add_masks(self) -> None:
        """
        Adds masks to the cameras in the current chunk.
        """
        self.doc.chunk.generateMasks(
            path=f"{self.cfg['asfm']['down_masks']}/{{filename}}_mask.png",
            masking_mode=ms.MaskingMode.MaskingModeFile,
            cameras=self.doc.chunk.cameras,
        )
        self.save_project()

    def detect_markers(self, chunk: int = 0, progress_callback: Callable = percentage_callback) -> None:
        """
        Detects 12 or 14-bit circular markers in the specified chunk.

        Args:
            chunk (int, optional): Chunk index. Defaults to 0.
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        log.info(f"Detecting markers: {self.markerbit}")
        self.doc.chunks[chunk].detectMarkers(
            target_type=self.markerbit,
            tolerance=50,
            filter_mask=False,
            inverted=False,
            noparity=False,
            maximum_residual=5,
            minimum_size=0,
            minimum_dist=5,
            cameras=self.doc.chunks[chunk].cameras,
            progress=progress_callback,
        )
        self.save_project()

    def import_reference(self, chunk: int = 0) -> None:
        """
        Imports reference points from a CSV file into the specified chunk.

        Args:
            chunk (int, optional): Chunk index. Defaults to 0.
        """
        self.doc.chunks[chunk].importReference(
            path=self.cfg['paths']['gcp_ref'],
            format=ms.ReferenceFormatCSV,
            columns="[n|x|y|z]",
            delimiter=";",
            group_delimiters=False,
            skip_rows=1,
            ignore_labels=False,
            create_markers=True,
            threshold=0.1,
            shutter_lag=0,
            crs=ms.CoordinateSystem(self.crs)
        )
        self.save_project()

    def export_camera_reference(self) -> None:
        """
        Exports the camera reference data to a CSV file.
        """
        reference = []
        for camera in self.doc.chunk.cameras:
            stats = CameraStats(camera).to_dict()
            calibration_params = self.camera_parameters(camera)
            stats.update(calibration_params)
            is_aligned = camera.transform is not None
            stats.update({"Alignment": is_aligned})
            reference.append(stats)

        dataframe = DataFrame(reference, "label")
        self.camera_reference = dataframe
        dataframe.to_csv(self.cfg['paths']['cam_ref'], header=True, index=False)

    def export_gcp_reference(self) -> None:
        """
        Exports the ground control points (GCP) reference data to a CSV file.
        """
        reference = []
        for marker in self.doc.chunk.markers:
            stats = MarkerStats(marker).to_dict()
            is_detected = len(marker.projections.items()) > 0
            stats.update({"Detected": is_detected})
            reference.append(stats)

        dataframe = DataFrame(reference, "label")
        self.gcp_reference = dataframe
        dataframe.to_csv(self.cfg['paths']['gcp_ref'], header=True, index=False)

    def optimize_cameras(self, progress_callback: Callable = percentage_callback) -> None:
        """
        Optimizes the cameras in the current chunk.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        optcam_cfg = self.cfg['asfm']['optimize_cameras_cfg']

        if optcam_cfg['disable_ref_cam']:
            n_cameras = len(self.doc.chunk.cameras)
            for i in range(n_cameras):
                self.doc.chunk.cameras[i].reference.enabled = False
        
        self.doc.chunk.optimizeCameras(
            fit_f=optcam_cfg['fit_f'],
            fit_cx=optcam_cfg['fit_cx'],
            fit_cy=optcam_cfg['fit_cy'],
            fit_b1=optcam_cfg['fit_b1'],
            fit_b2=optcam_cfg['fit_b2'],
            fit_k1=optcam_cfg['fit_k1'],
            fit_k2=optcam_cfg['fit_k2'],
            fit_k3=optcam_cfg['fit_k3'],
            fit_k4=optcam_cfg['fit_k4'],
            fit_p1=optcam_cfg['fit_p1'],
            fit_p2=optcam_cfg['fit_p2'],
            fit_corrections=optcam_cfg['fit_corrections'],
            adaptive_fitting=optcam_cfg['adaptive_fitting'],
            tiepoint_covariance=optcam_cfg['tiepoint_covariance'],
            progress=progress_callback,
        )

        self.save_project()

    def get_unaligned_cameras(self, chunk: int = 0) -> List[ms.Camera]:
        """
        Returns a list of unaligned cameras in the specified chunk.

        Args:
            chunk (int, optional): Chunk index. Defaults to 0.

        Returns:
            List[ms.Camera]: List of unaligned cameras.
        """
        unaligned_cameras = [
            camera
            for camera in self.doc.chunks[chunk].cameras
            if camera.transform is None
        ]
        return unaligned_cameras

    def reset_region(self) -> bool:
        """
        Resets the region to be larger than the points, preventing clipping when saving.

        Returns:
            bool: True if the region is reset successfully.
        """
        self.doc.chunk.resetRegion()
        region_dims = self.doc.chunk.region.size
        region_dims[2] *= 3
        self.doc.chunk.region.size = region_dims

        return True

    def match_photos(
        self,
        progress_callback: Callable = percentage_callback,
        chunk: int = 0,
        reference_preselection=ms.ReferencePreselectionSource
    ) -> None:
        """
        Matches photos in the specified chunk using the provided settings.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
            chunk (int, optional): Chunk index. Defaults to 0.
            reference_preselection (optional): Reference preselection mode. Defaults to ms.ReferencePreselectionSource.
        """
        log.info("Matching photos")

        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus
        match_cfg = self.cfg['asfm']['match_photos']
        
        self.doc.chunks[chunk].matchPhotos(
            downscale=match_cfg['downscale'],
            generic_preselection=match_cfg['generic_preselection'],
            reference_preselection=match_cfg['reference_preselection'],
            reference_preselection_mode=reference_preselection,
            filter_mask=match_cfg['filter_mask'],
            mask_tiepoints=match_cfg['mask_tiepoints'],
            filter_stationary_points=match_cfg['filter_stationary_points'],
            keypoint_limit=match_cfg['keypoint_limit'],
            keypoint_limit_per_mpx=match_cfg['keypoint_limit_per_mpx'],
            tiepoint_limit=match_cfg['tiepoint_limit'],
            keep_keypoints=match_cfg['keep_keypoints'],
            cameras=self.doc.chunks[chunk].cameras,
            guided_matching=match_cfg['guided_matching'],
            reset_matches=match_cfg['reset_matches'],
            subdivide_task=match_cfg['subdivide_task'],
            workitem_size_cameras=match_cfg['workitem_size_cameras'],
            workitem_size_pairs=match_cfg['workitem_size_pairs'],
            max_workgroup_size=match_cfg['max_workgroup_size'],
            progress=progress_callback,
        )

        ms.app.cpu_enable = bool(ms.app.gpu_mask)
        self.save_project()

    def align_photos(
        self,
        progress_callback: Callable = percentage_callback,
        chunk: int = 0,
        correct: bool = False
    ) -> None:
        """
        Aligns photos in the specified chunk and optionally corrects unaligned cameras.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
            chunk (int, optional): Chunk index. Defaults to 0.
            correct (bool, optional): Whether to correct unaligned cameras. Defaults to False.
        """
        log.info("Aligning photos")
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        align_cfg = self.cfg['asfm']['align_photos']

        self.doc.chunks[chunk].alignCameras(
            cameras=self.doc.chunks[chunk].cameras,
            min_image=align_cfg['min_image'],
            adaptive_fitting=align_cfg['adaptive_fitting'],
            reset_alignment=align_cfg['reset_alignment'],
            subdivide_task=align_cfg['subdivide_task'],
            progress=progress_callback
        )
        ms.app.cpu_enable = bool(ms.app.gpu_mask)
        self.save_project()

        unaligned_cameras = self.get_unaligned_cameras(chunk)
        if unaligned_cameras:
            log.warning(f"Found {len(unaligned_cameras)} unaligned cameras.")
            if correct:
                self._correct_unaligned_cameras(unaligned_cameras, chunk, progress_callback)

        self._remove_duplicate_and_unaligned_cameras()
        self.reset_region()
        self.save_project()

    def _correct_unaligned_cameras(
        self,
        unaligned_cameras: List[ms.Camera],
        chunk: int,
        progress_callback: Callable
    ) -> None:
        """
        Attempts to correct unaligned cameras by reprocessing them.

        Args:
            unaligned_cameras (List[ms.Camera]): List of unaligned cameras.
            chunk (int): Chunk index.
            progress_callback (Callable): Progress callback function.
        """
        log.warning("Correction enabled. Checking for unaligned cameras.")
        
        if len(unaligned_cameras) < 1:
            log.info("Not enough unaligned cameras to perform alignment, skipping.")
            return

        log.info("Attempting to align unaligned cameras.")
        new_chunk = self.doc.addChunk()
        photos = [camera.photo.path for camera in unaligned_cameras]

        new_chunk.addPhotos(photos)

        self.detect_markers(chunk=len(self.doc.chunks) - 1)
        self.import_reference(chunk=len(self.doc.chunks) - 1)

        log.info("Matching and Aligning photos again.")
        self.match_photos(chunk=len(self.doc.chunks) - 1)
        self.align_photos(chunk=len(self.doc.chunks) - 1, correct=False)

        log.info("Merging Chunks.")
        self.doc.mergeChunks(chunks=[chunk, len(self.doc.chunks) - 1], merge_markers=True, progress=progress_callback)
        log.info("Setting active chunk.")
        self.doc.chunk = self.doc.chunks[-1]

    def _remove_duplicate_and_unaligned_cameras(self) -> None:
        """
        Removes duplicate aligned cameras and unaligned cameras from the chunk.
        """
        unique_aligned_cameras = set()
        cameras_to_remove = []

        for camera in self.doc.chunk.cameras:
            if camera.transform is None:
                cameras_to_remove.append(camera)
            elif camera.label in unique_aligned_cameras:
                cameras_to_remove.append(camera)
            else:
                unique_aligned_cameras.add(camera.label)

        for camera in cameras_to_remove:
            self.doc.chunk.remove(camera)

        log.info(f"Unaligned cameras in current chunk: {len(self.get_unaligned_cameras())}")
        log.info(f"Final number of cameras in current chunk after realignment: {len(self.doc.chunk.cameras)}")

    def build_depth_map(self, progress_callback: Callable = percentage_callback) -> None:
        """
        Builds depth maps for the current chunk.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus
        
        log.info(f"Number of cameras in chunk at depth map: {len(self.doc.chunk.cameras)}")
        depth_cfg = self.cfg['asfm']['depth_map']

        self.doc.chunk.buildDepthMaps(
            downscale=depth_cfg['downscale'],
            filter_mode=getattr(ms, depth_cfg['filter_mode']),
            cameras=self.doc.chunk.cameras,
            reuse_depth=depth_cfg['reuse_depth'],
            max_neighbors=depth_cfg['max_neighbors'],
            subdivide_task=depth_cfg['subdivide_task'],
            workitem_size_cameras=depth_cfg['workitem_size_cameras'],
            max_workgroup_size=depth_cfg['max_workgroup_size'],
            progress=progress_callback,
        )

        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        
        if depth_cfg['autosave']:
            self.save_project()

    def build_dense_cloud(self, progress_callback: Callable = percentage_callback) -> None:
        """
        Builds a dense point cloud for the current chunk.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        if self.doc.chunk.depth_maps is None:
            self.build_depth_map()

        ptcld_cfg = self.cfg['asfm']['point_cloud']
        self.doc.chunk.buildPointCloud(
            point_colors=ptcld_cfg['point_colors'],
            point_confidence=ptcld_cfg['point_confidence'],
            keep_depth=ptcld_cfg['keep_depth'],
            max_neighbors=ptcld_cfg['max_neighbors'],
            uniform_sampling=ptcld_cfg['uniform_sampling'],
            subdivide_task=ptcld_cfg['subdivide_task'],
            workitem_size_cameras=ptcld_cfg['workitem_size_cameras'],
            max_workgroup_size=ptcld_cfg['max_workgroup_size'],
            progress=progress_callback,
        )
        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        if ptcld_cfg['autosave']:
            self.save_project()

    def build_model(self, progress_callback: Callable = percentage_callback) -> None:
        """
        Builds a 3D model from the dense point cloud.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        bldmod_cfg = self.cfg['asfm']['model']
        
        self.doc.chunk.buildModel(
            surface_type=getattr(ms, bldmod_cfg['surface_type']),
            interpolation=getattr(ms, bldmod_cfg['interpolation']),
            face_count=getattr(ms, bldmod_cfg['face_count']),
            source_data=getattr(ms, bldmod_cfg['source_data']),
            vertex_colors=bldmod_cfg['vertex_colors'],
            vertex_confidence=bldmod_cfg['vertex_confidence'],
            volumetric_masks=bldmod_cfg['volumetric_masks'],
            keep_depth=bldmod_cfg['keep_depth'],
            trimming_radius=bldmod_cfg['trimming_radius'],
            subdivide_task=bldmod_cfg['subdivide_task'],
            workitem_size_cameras=bldmod_cfg['workitem_size_cameras'],
            max_workgroup_size=bldmod_cfg['max_workgroup_size'],
            progress=progress_callback,
        )
        
        self.doc.chunk.model.closeHoles(level=100)
        self.save_project()

    def build_dem(self, progress_callback: Callable = percentage_callback) -> None:
        """
        Builds a digital elevation model (DEM) from the dense point cloud.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        if self.doc.chunk.point_cloud is None:
            self.build_dense_cloud()

        dem_cfg = self.cfg['asfm']['dem']
        
        self.doc.chunk.buildDem(
            source_data=getattr(ms, dem_cfg['source_data']),
            interpolation=getattr(ms, dem_cfg['interpolation']),
            flip_x=dem_cfg['flip_x'],
            flip_y=dem_cfg['flip_y'],
            flip_z=dem_cfg['flip_z'],
            resolution=dem_cfg['resolution'],
            subdivide_task=dem_cfg['subdivide_task'],
            workitem_size_tiles=dem_cfg['workitem_size_tiles'],
            max_workgroup_size=dem_cfg['max_workgroup_size'],
            progress=progress_callback,
        )
        
        if dem_cfg['autosave']:
            self.save_project()

        if dem_cfg['export']:
            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True
            kwargs = {"image_compression": image_compression}

            self.doc.chunk.exportRaster(
                path=self.cfg['paths']['dem_path'],
                image_format=ms.ImageFormatTIFF,
                source_data=ms.ElevationData,
                progress=progress_callback,
                **kwargs,
            )

    def build_orthomosaic(self, progress_callback: Callable = percentage_callback) -> None:
        """
        Builds an orthomosaic from the model or DEM.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        ortho_cfg = self.cfg['asfm']['ortho']

        self.doc.chunk.buildOrthomosaic(
            surface_data=getattr(ms, ortho_cfg['surface_data']),
            blending_mode=getattr(ms, ortho_cfg['blending_mode']),
            fill_holes=ortho_cfg['fill_holes'],
            ghosting_filter=ortho_cfg['ghosting_filter'],
            cull_faces=ortho_cfg['cull_faces'],
            refine_seamlines=ortho_cfg['refine_seamlines'],
            resolution=ortho_cfg['resolution'],
            resolution_x=ortho_cfg['resolution_x'],
            resolution_y=ortho_cfg['resolution_y'],
            flip_x=ortho_cfg['flip_x'],
            flip_y=ortho_cfg['flip_y'],
            flip_z=ortho_cfg['flip_z'],
            subdivide_task=ortho_cfg['subdivide_task'],
            workitem_size_cameras=ortho_cfg['workitem_size_cameras'],
            workitem_size_tiles=ortho_cfg['workitem_size_tiles'],
            max_workgroup_size=ortho_cfg['max_workgroup_size'],
            progress=progress_callback,
        )
        if ortho_cfg['autosave']:
            self.save_project()

        if ortho_cfg['export']:
            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True

            self.doc.chunk.exportRaster(
                path=self.cfg['paths']['ortho_path'],
                image_format=ms.ImageFormatTIFF,
                source_data=ms.OrthomosaicData,
                progress=progress_callback,
                image_compression=image_compression,
            )

    def camera_parameters(self, camera: ms.Camera) -> Dict[str, Optional[float]]:
        """
        Retrieves the camera parameters for the specified camera.

        Args:
            camera (ms.Camera): Camera object.

        Returns:
            Dict[str, Optional[float]]: Dictionary of camera parameters.
        """
        row = dict()
        row["f"] = camera.calibration.f  # Focal length in pixels
        row["cx"] = camera.calibration.cx
        row["cy"] = camera.calibration.cy
        row["k1"] = camera.calibration.k1
        row["k2"] = camera.calibration.k2
        row["k3"] = camera.calibration.k3
        row["k4"] = camera.calibration.k4
        row["p1"] = camera.calibration.p1
        row["p2"] = camera.calibration.p2
        row["b1"] = camera.calibration.b1
        row["b2"] = camera.calibration.b2
        row["pixel_height"] = camera.sensor.pixel_height
        row["pixel_width"] = camera.sensor.pixel_width

        return row

    def export_stats(self) -> None:
        """
        Exports statistical data, including the percentage of aligned cameras and detected markers.
        """
        total_cameras = len(self.doc.chunk.cameras)
        aligned_cameras = sum(row["Alignment"] for row in self.camera_reference.content_dict)
        percentage_aligned_cameras = aligned_cameras / total_cameras

        total_gcps = len(self.gcp_reference)
        detected_gcps = sum(row["Detected"] for row in self.gcp_reference.content_dict)
        percentage_detected_gcps = detected_gcps / max(1, total_gcps)

        dataframe = DataFrame(
            [
                {
                    "Total_Cameras": total_cameras,
                    "Aligned_Cameras": aligned_cameras,
                    "Percentage_Aligned_Cameras": percentage_aligned_cameras,
                    "Total_GCPs": total_gcps,
                    "Detected_GCPs": detected_gcps,
                    "Percentage_Detected_GCPs": percentage_detected_gcps,
                }
            ],
            "Total_Cameras",
        )
        self.error_statistics = dataframe
        dataframe.to_csv(self.cfg['paths']['err_ref'], header=True, index=False)

    def camera_fov(self) -> None:
        """
        Calculates the field of view (FOV) for each camera and exports the data to a CSV file.
        """
        rows = []
        row_template = {
            "label": "",
            "top_left_x": "",
            "top_left_y": "",
            "bottom_left_x": "",
            "bottom_left_y": "",
            "bottom_right_x": "",
            "bottom_right_y": "",
            "top_right_x": "",
            "top_right_y": "",
            "height": "",
            "width": "",
        }

        for camera in self.doc.chunk.cameras:
            row = deepcopy(row_template)
            calculate_fov = True

            row["label"] = camera.label

            pixel_height = camera.sensor.pixel_height
            if pixel_height is None:
                log.warning(f"pixel_height missing for camera {camera.label}, skipping FOV calculation.")
                calculate_fov = False

            pixel_width = camera.sensor.pixel_width
            if pixel_width is None:
                log.warning(f"pixel_width missing for camera {camera.label}, skipping FOV calculation.")
                calculate_fov = False

            height = camera.sensor.height
            if height is None:
                log.warning(f"height missing for camera {camera.label}, skipping FOV calculation.")
                calculate_fov = False

            width = camera.sensor.width
            if width is None:
                log.warning(f"width missing for camera {camera.label}, skipping FOV calculation.")
                calculate_fov = False

            f = camera.calibration.f
            if f is None:
                log.warning(f"f missing for camera {camera.label}, skipping FOV calculation.")
                calculate_fov = False

            if calculate_fov:
                # Convert focal length and image dimensions to real-world units
                f_height = f * pixel_height
                f_width = f * pixel_width
                image_height = height * pixel_height
                image_width = width * pixel_width

                # Find the actual object height and width
                camera_height = self.camera_reference.retrieve(camera.label, "Estimated_Z")

                if not camera_height:
                    continue
                object_half_height = find_object_dimension(f_height, image_height / 2, camera_height)
                object_half_width = find_object_dimension(f_width, image_width / 2, camera_height)

                row["height"] = 2.0 * object_half_height
                row["width"] = 2.0 * object_half_width

                # Find the field of view coordinates in the rotated
                yaw_angle = self.camera_reference.retrieve(camera.label, "Estimated_Yaw")

                center_x = self.camera_reference.retrieve(camera.label, "Estimated_X")
                center_y = self.camera_reference.retrieve(camera.label, "Estimated_Y")
                if not yaw_angle or not center_x or not center_y:
                    continue

                center_coords = [center_x, center_y]

                (
                    top_left_x,
                    top_left_y,
                    bottom_left_x,
                    bottom_left_y,
                    bottom_right_x,
                    bottom_right_y,
                    top_right_x,
                    top_right_y,
                ) = field_of_view(center_coords, object_half_width, object_half_height, yaw_angle)

                row["top_left_x"], row["top_left_y"] = top_left_x, top_left_y
                row["bottom_left_x"], row["bottom_left_y"] = bottom_left_x, bottom_left_y
                row["bottom_right_x"], row["bottom_right_y"] = bottom_right_x, bottom_right_y
                row["top_right_x"], row["top_right_y"] = top_right_x, top_right_y

            rows.append(row)

        df = DataFrame(rows, "label")
        df.to_csv(self.cfg['paths']['fov_ref'], index=False, header=True)

    def export_report(self, progress_callback: Callable = percentage_callback) -> None:
        """
        Exports a project report in PDF format.

        Args:
            progress_callback (Callable, optional): Progress callback function. Defaults to percentage_callback.
        """
        self.doc.chunk.exportReport(
            path=f"{self.cfg['paths']['project_asfm']}/{self.cfg['project_name']}.pdf",
            title=self.cfg['project_name'],
            description="report",
            font_size=12,
            page_numbers=True,
            include_system_info=True,
            progress=progress_callback,
        )
