# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Vertex AI Pipeline implementation
- Use pathlib with all paths instead

## [0.28.2] - 2022-10-31
### Fixed
- Fix CLI tool for GEE auth
- Fix Sandbox notebook

## [0.28.1] - 2022-10-26
### Fixed
- Fix CLI tool output path

## [0.28.0] - 2022-10-26
### Added
- Makes script OceanCode complaint

### Fixed
- Dockerfile

## [0.27.4] - 2022-10-23
### Added
- Jupyter-lab out-of-the-box from Docker image

### Fixed
- Dockerfile

## [0.27.3] - 2022-10-23
### Fixed
- README.md and CHANGELOG -> .md

## [0.27.2] - 2022-10-23
### Fixed
- Dockerfile ending after run

## [0.27.0] - 2022-10-23
### Added
- Command line tool for running detections
- Versioning control (tags, setup.py, version.py)
- Black formatting
- Docker image

### Fixed
- Python package structure best practices

## [0.26.0]
### Added
- Tests with NDVI and FAI only
- Created yearly outputs

## [0.25.0]
### Added
- SABI Index
- Isolation Forest support

## [0.24.0]
### Added
- Random Forest support for RandomizedSearchCV
- New ROIs were selected to validate the algorithm with in situ data
- New function for outlier removal
- Support for bigger study areas
- Support for Deep Learning with TensorFlow AutoEncoder

## [0.23.2]
### Added
- Built-in image caching system for increased performance

## [0.23.1]
### Fixed
- Try except when extracting pixels from images
- Trend removal correction

## [0.23.0]
### Added
- Merge test between GridSearch and RandomizedSearch
- Removal of the Outlier parameter - set to False, that is, remove
- Enabled Modis satellite

### Fixed
- Image error control, if not found, including Modis water mask
- Corrections in the generation of the recurrences graph

### Removed
- Poly kernel (optimization)

## [0.22.2]
### Added
- Modification to support mosaic of images of the same date with cut (Sentinel)

## [0.22.1]
### Added
- Creating script for cloud separation in json, correction in the display of maps

## [0.22.0]
### Added
- Trend was removed using the medium image and subtracting its pixels
from the time series images, adding the use of randomizedsearchcv

## [0.21.0-2]
### Added
- Mean and standard deviation were tested with STD, STD / 2 and STD + (STD / 2)
in versions 21.0, 21.1 and 21.2 respectively
