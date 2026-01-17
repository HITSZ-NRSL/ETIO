# ETIO


This repository contains code for Edge-based monocular thermal-inertial odometry in visually degraded environments.

### 1. Dependency
- Basic libraries for common SLAM (e.g. openCV 4, pcl-1.10, ceres-solver-1.14 etc.)

### 2. Run Examples
```
mkdir -p YOUR_NAME_OF_WORKSPACE/src
cd YOUR_NAME_OF_WORKSPACE/src
git clone https://github.com/HITSZ-NRSL/ETIO.git
cd ..
```
`catkin_make` or `catkin build`

```
roslaunch mytool mytool.launch 
roslaunch etio tau2.launch
```
For validation, the public datasets are provided: [dowload link](https://www.autonomousrobotslab.com/ktio.html) 

### 3.  Citation
If you use our work, please cite:
```
@ARTICLE{10048516,
  author={Wang, Yu and Chen, Haoyao and Liu, Yufeng and Zhang, Shiwu},
  journal={IEEE Robotics and Automation Letters}, 
  title={Edge-Based Monocular Thermal-Inertial Odometry in Visually Degraded Environments}, 
  year={2023},
  volume={8},
  number={4},
  pages={2078-2085},
  keywords={Image edge detection;Cameras;Feature extraction;Visualization;State estimation;Robustness;Noise measurement;Localization;search and rescue robots;visual-inertial SLAM},
  doi={10.1109/LRA.2023.3246381}}
```
## LICENSE
The source code is released under GPLv3 license.
