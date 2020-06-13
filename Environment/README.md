## Autonomous Environment Mapping Robot
The full script for the simulation of the autonomous environment mapping robot in the Pybullet Simulator has been given in this directory as Object_Detection_Final_beta.py. All the parts of the project - Object Detection using Deep Learning, Depth Map using StereoCamera and OpenCV along with autonomous control of the robot, all have been compiled together to give a running script. For running the script, this directory must be downloaded as it is. 

### Environment
The environment is a 20 by 20 part of the plane. The arena has been prepared in such a manner that it would take at least 2-3 passes of the husky to be able to ‘see’ the other husky in its camera. Two each of ‘husky’, ‘teddy_large’, ‘humanoid’, ‘duck_vhacd’ and ‘cube’, and three each of ‘r2d2’ and ‘sphere2red’ were used, all of which are inbuilt objects available in the pybullet_data package. All the corresponding URDFs along with their .obj and .mtl files are given in this directory.
<p align="center">
 <img  width="400" height="400" src="https://github.com/Terabyte17/Autonomous-Room-Mapping-Robot/blob/master/Grid%20Media/Grid.png">
</p>


### Autonomous Run
The robot starts moving autonomously after the key 'h' has been pressed at the start of the simulation. It takes around 9 mins for the car 'husky' to reach and detect the other 'husky'. Once it detects the car, an image is shown. For exiting the simulation after the car has been detected, any button can be pressed.

#### Make sure that you have installed the latest version of ImageAI and the relevant version of tensorflow, before running this script.
