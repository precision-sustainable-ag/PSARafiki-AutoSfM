import sys
from pathlib import Path
import getpass
import logging

# Add the src directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import hydra
from omegaconf import DictConfig
from omegaconf import MISSING, OmegaConf  # Do not confuse with dataclass.MISSING
from hydra.utils import get_method

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    whoami = getpass.getuser()
    log.info(f"Running {cfg.task} as {whoami}")

    try:
        task = get_method(f"{cfg.task}.main")
        task(cfg)

    except Exception as e:
        log.exception("Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
