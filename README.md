# Geometry Perception Utils

This repository holds the code utilities for several purposes. It contents common functions for i/o data reading, draw in images, visualization of point clouds, camera projections, different camera models,... etc. 

## Used in the following projects
* [Ray-Casting-MLC (ECCV2024) ](https://enriquesolarte.github.io/ray-casting-mlc/)
 & [MVL-DATASETS](https://huggingface.co/datasets/EnriqueSolarte/mvl_datasets)

* [360-MLC (NeurIPS 2022)](https://enriquesolarte.github.io/360-mlc/)

* [360-DFPE (RA-L 2022)](https://enriquesolarte.github.io/360-dfpe/)

* [Robust 360-8PA (ICRA 2021)](https://enriquesolarte.github.io/robust_360_8pa/)


## News and updates
- **2025-04-16**: Update @latest to @v1.0.4
## Installation

### Create a virtual environment
```sh 
conda create -n utils python=3.9
conda activate utils
```

### Install the package from the repository
```sh
pip install git+https://github.com/EnriqueSolarte/geometry_perception_utils.git@latest
```

### For installing this package in dev mode (for development)
```sh 
git clone https://github.com/EnriqueSolarte/geometry_perception_utils.git@latest
cd geometry_perception_utils
pip install -e .
```
