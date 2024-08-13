import glob
import logging
import os
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import piexif
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up the logger
log = logging.getLogger(__name__)

def remove_missing_data(cfg) -> None:
    """
    Removes missing downscaled images or masks that do not have a corresponding file.

    Args:
        cfg (DictOmega): Hydra configuration object.
    """
    imgs = glob.glob(os.path.join(cfg.asfm.down_photos, "*.jpg"))
    masks = glob.glob(os.path.join(cfg.asfm.down_masks, "*.png"))

    imgs = [Path(img).stem for img in imgs]
    masks = [Path(mask).stem.replace("_mask", "") for mask in masks]

    # Find the missing and additional elements in masks
    miss = list(set(imgs).difference(masks))
    add = list(set(masks).difference(imgs))

    if (len(miss) == 0) and (len(add) == 0):
        return
    elif len(miss) > len(add):
        # More photos than masks. Remove extra photos
        log.warning(f"More photos than masks. Removing {len(miss)} extra downscaled photos")
        for img in miss:
            Path(cfg.asfm.down_photos, img + ".jpg").unlink()
    elif len(add) > len(miss):
        # More masks than photos. Remove extra masks
        log.warning(f"More masks than photos. Removing {len(add)} extra downscaled masks")
        for mask in add:
            Path(cfg.asfm.down_masks, mask + "_mask.png").unlink()

def resize_and_save(data: dict) -> None:
    """
    Resizes and saves an image or mask according to the given scale.

    Args:
        data (dict): Dictionary containing image source, destination, scale, and mask flag.
    """
    image_src = data["image_src"]
    image_dst = data["image_dst"]
    scale = data["scale"]
    masks = data["masks"]

    assert 0.0 < scale <= 1.0, "scale should be between (0, 1]."

    try:
        image = Image.open(image_src)
        width, height = image.size
        scaled_width, scaled_height = int(ceil(width * scale)), int(ceil(height * scale))
        kwargs = {}

        if masks:
            # Save mask without additional processing
            resized_image = image.resize((scaled_width, scaled_height))
            resized_image.save(image_dst, quality=95, **kwargs)
        else:
            # Attempt to preserve EXIF data during resizing
            try:
                exif_data = piexif.load(image.info["exif"])
                exif_bytes = piexif.dump(exif_data)
                kwargs["exif"] = exif_bytes
            except KeyError:
                log.warning("EXIF data not found, resizing without EXIF data.")

            resized_image = image.resize((scaled_width, scaled_height))
            resized_image.save(image_dst, quality=95, **kwargs)

    except (IOError, SyntaxError) as e:
        log.error(f"Bad file: {image_src}. Error: {e}")

def glob_multiple_extensions(directory: Union[str, Path], extensions: List[str]) -> List[Path]:
    """
    Glob files with multiple extensions using pathlib.

    Args:
        directory (Union[str, Path]): Path to the directory to search in.
        extensions (List[str]): List of file extensions to search for.

    Returns:
        List[Path]: List of Paths matching the given extensions.
    """
    if isinstance(directory, str):
        directory = Path(directory)

    # Ensure extensions are formatted correctly
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

    # Gather all files matching the extensions
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
    
    return files

def resize_photo_diretory(cfg) -> None:
    """
    Resizes photos in a directory according to the configuration settings.

    Args:
        cfg (DictOmega): Hydra configuration object.
    """
    base_path = cfg["paths"]["input_images"]
    save_dir = cfg["paths"]["down_photos"]

    file_extensions = ['jpg', 'JPG', 'png', 'PNG', 'bmp', 'BMP']
    files = glob_multiple_extensions(base_path, file_extensions)

    num_files = len(files)
    log.info(f"Processing {num_files} files.")

    data = [
        {
            "image_src": src,
            "image_dst": Path(save_dir, src.name),
            "scale": cfg["asfm"]["downscale"]["factor"],
            "masks": False,
        }
        for src in files
    ]

    # Determine the number of processes for parallel processing
    num_processes = int(len(os.sched_getaffinity(0)) / cfg.cpu_denominator)

    try:
        with Pool(num_processes) as pool:
            for i, _ in enumerate(pool.imap_unordered(resize_and_save, data), 1):
                print(f"Progress: {i}/{num_files} images resized")
    except KeyboardInterrupt:
        log.info("Interrupted by user, terminating...")
        pool.terminate()
    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
        log.info("Completed resizing images.")
