import sys
from pathlib import Path
import getpass
import logging

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

# Set up logging
log = logging.getLogger(__name__)

# Add the src directory to the PYTHONPATH for easier imports
sys.path.append(str(Path(__file__).resolve().parent / "src"))

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for running a Hydra-configured task.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    cfg = OmegaConf.create(cfg)
    whoami = getpass.getuser()
    log.info(f"Running {cfg.task} as {whoami}")

    try:
        # Dynamically retrieve and execute the specified task method
        task = get_method(f"{cfg.task}.main")
        task(cfg)
    except Exception as e:
        log.exception("Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
