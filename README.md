# PSARafiki-AutoSfM

This repo is a toolkit for automating the Structure from Motion (SfM) image processing pipeline using Agisoft Metashape. It includes scripts and utilities that facilitate the creation of 3D models, orthomosaics, digital elevation models (DEMs), and other outputs essential for photogrammetry and remote sensing applications.
Key Features

    Automated Processing Pipeline: Streamline the workflow from image input to final 3D model generation, significantly reducing manual intervention.
    Pixel-to-Real-World Coordinate Mapping: Convert pixel-based image coordinates into accurate 3D real-world coordinates, enabling precise feature identification and analysis across spatial plots.
    Overlapping Images Required: Designed to work with image sets that have at least 75% overlap both side-to-side and front-to-back, ensuring robust and accurate reconstruction.

Requirements

    Agisoft Metashape License: A valid license for Agisoft Metashape is required to run the processing scripts.
    Compute Resources: Adequate computing power (CPU, GPU, and memory) to handle large datasets and complex processing tasks.
    Python 3.11: The scripts are developed and tested using Python 3.11.
    Image Data: A set of overlapping images with a minimum of 75% side and front overlap for optimal 3D reconstruction.