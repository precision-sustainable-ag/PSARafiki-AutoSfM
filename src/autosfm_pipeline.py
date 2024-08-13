import logging
import signal
import sys
from omegaconf import DictConfig

from utils.config_utils import autosfm_present, create_config
from utils.metashape_utils import SfM
from utils.resize import resize_photo_diretory

# Set up the logger
log = logging.getLogger(__name__)

class SfMProcessingError(Exception):
    """Custom exception for errors during SfM processing."""
    pass

def main(cfg: DictConfig) -> None:
    """
    Main function to run the SfM pipeline based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration dictionary for the SfM process.
    """
    def sigint_handler(signum: int, frame) -> None:
        """
        Signal handler for SIGINT.

        Args:
            signum (int): Signal number.
            frame: Current stack frame.
        """
        print("\nPython SIGINT detected. Exiting.\n")
        sys.exit(1)

    def sigterm_handler(signum: int, frame) -> None:
        """
        Signal handler for SIGTERM.

        Args:
            signum (int): Signal number.
            frame: Current stack frame.
        """
        print("\nPython SIGTERM detected. Exiting.\n")
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        # Setup configuration
        cfg = create_config(cfg)

        # Check if autosfm has already been run
        if cfg.asfm.check_for_asfm:
            log.info("Checking for autosfm contents")
            if autosfm_present(cfg):
                log.info("Autosfm has already been run. All contents are available. Moving to next process.")
                return

        # Resize images and masks if needed
        if cfg.asfm.resize_photos:
            if cfg["asfm"]["downscale"]["enabled"]:
                log.info("Resizing images")
                resize_photo_diretory(cfg)

        # Initialize SfM pipeline
        log.info("Initializing SfM")
        pipeline = SfM(cfg)

        # Add photos and masks to the Metashape project
        if cfg.asfm.add_photos_and_masks:
            log.info("Adding photos")
            pipeline.add_photos()
            if cfg["asfm"]["use_masking"]:
                pipeline.add_masks()

        # Detect markers in the images
        if cfg.asfm.detect_markers:
            log.info("Detecting markers")
            pipeline.detect_markers()

        # Import marker locations (references)
        if cfg.asfm.import_references:
            log.info("Importing references")
            pipeline.import_reference()

        # Match photos to find common points
        if cfg.asfm.match:
            log.info("Matching photos")
            pipeline.match_photos()

        # Align photos to reconstruct camera positions
        if cfg.asfm.align:
            log.info("Aligning photos")
            pipeline.align_photos(correct=True)

        # Optimize camera alignment
        if cfg.asfm.optimize_cameras:
            log.info("Optimizing cameras")
            pipeline.optimize_cameras()

        # Export GCP and camera reference data, and error statistics
        if cfg.asfm.export_gcp_camref_err:
            log.info("Exporting GCP reference")
            pipeline.export_gcp_reference()

            log.info("Exporting camera reference")
            pipeline.export_camera_reference()

            log.info("Exporting error stats")
            pipeline.export_stats()

        # Elective processes

        # Build depth map if configured
        if cfg.asfm.build_depth:
            log.info("Building depth maps")
            pipeline.build_depth_map()

        # Build dense point cloud if configured
        if cfg.asfm.build_point_cloud:
            log.info("Building dense point cloud")
            pipeline.build_dense_cloud()

        # Build 3D model if configured
        if cfg.asfm.build_model and cfg["asfm"]["model"]["enabled"]:
            log.info("Building model")
            pipeline.build_model()

        # Build Digital Elevation Model (DEM) if configured
        if cfg.asfm.build_dem:
            log.info("Building DEM")
            pipeline.build_dem()

        # Build orthomosaic if configured
        if cfg.asfm.build_ortho:
            log.info("Building orthomosaic")
            pipeline.build_orthomosaic()

        # Export Field of View (FOV) data if configured
        if cfg.asfm.export_fov:
            log.info("Exporting camera FOV information")
            pipeline.camera_fov()

        # Export final report if configured
        if cfg.asfm.export_report:
            log.info("Exporting report")
            pipeline.export_report()

        log.info("AutoSfM Complete")

    except SfMProcessingError as e:
        log.error(f"An error occurred during SfM processing: {e}")
        sys.exit(1)
    except Exception as e:
        log.exception("An unexpected error occurred.")
        sys.exit(1)

# Uncomment and modify the following lines for script execution
# if __name__ == "__main__":
#     try:
#         # Example configuration loading
#         config = DictConfig({"asfm": {}})  # Replace with actual configuration loading
#         main(config)
#     except Exception as e:
#         log.exception("An error occurred in the main execution.")
#         sys.exit(1)
