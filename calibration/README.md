## Calibration of cameras to create depth map

The various files are explained below:-

Chessboard.urdf was created manually to simulate a chessboard in order to calibrate the cameras that were in PyBullet.
capture_calib.py is used to take images of the chessboard and save them in capture\left and capture\right folders respectively.
calibrate.py uses the captured images to first calibrate each camera individually and then together to work as stereo cameras.
depth.py produces a sample depth map from two input images.
calibration.npz stores the remapping matrices and data obtined from calibration.py and used in depth.py
