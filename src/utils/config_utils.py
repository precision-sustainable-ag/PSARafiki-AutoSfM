from pathlib import Path
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

def make_dir(path: Path) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        path (Path): Path of the directory to create.
    """
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)

def make_autosfm_dirs(cfg: DictConfig) -> None:
    """
    Creates necessary directories for the AutoSfM process.

    Args:
        cfg (DictConfig): Configuration dictionary containing directory paths.
    """
    make_dir(Path(cfg.paths.project_asfm))
    make_dir(Path(cfg.paths.down_photos))
    make_dir(Path(cfg.paths.down_masks))
    make_dir(Path(cfg.paths.refs))
    make_dir(Path(cfg.paths.orthodir))
    make_dir(Path(cfg.paths.demdir))

def create_config(cfg: DictConfig) -> DictConfig:
    """
    Creates and initializes the configuration for the SfM process.

    Args:
        cfg (DictConfig): Initial configuration dictionary.

    Returns:
        DictConfig: Updated configuration dictionary.
    """
    # Load Metashape key from YAML file
    # cfg.metashape_key = yaml.safe_load(cfg.pipeline_keys)
    
    # Create necessary directories
    make_autosfm_dirs(cfg)
    
    return cfg

def autosfm_present(cfg: DictConfig) -> bool:
    """
    Checks if the expected outputs of the SfM processing pipeline are present.

    Args:
        cfg (Dict): The configuration dictionary.

    Returns:
        bool: True if all expected outputs are present, False otherwise.
    """
    # Define expected output paths based on configuration
    expected_outputs = [
        Path(cfg["paths"]["project_asfm"]),
        Path(cfg["paths"]["down_photos"]),
        Path(cfg["paths"]["cam_ref"]),
        Path(cfg["paths"]["gcp_ref"]),
        Path(cfg["paths"]["err_ref"]),
        Path(cfg["paths"]["fov_ref"]),
        Path(cfg["paths"]["dem_path"]),
        Path(cfg["paths"]["ortho_path"]),
        Path(cfg["paths"]["project_asfm"], f"{cfg['project_name']}.pdf")
    ]

    # Check if masking is enabled and add mask output path
    if cfg["asfm"]["use_masking"]:
        expected_outputs.append(Path(cfg["paths"]["down_masks"]))

    # Use list comprehension to find missing paths
    missing_outputs = [str(path) for path in expected_outputs if not path.exists()]

    # Log and return based on missing outputs
    if missing_outputs:
        log.warning("Some expected outputs are missing: " + ", ".join(missing_outputs))
        return False

    log.info("All expected outputs are present.")
    return True
