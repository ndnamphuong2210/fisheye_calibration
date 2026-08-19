Fisheye lens calibration is made based on Kannala & Brandt's generic camera model. Although OpenCV fisheye lens calibration is also implemented based on the same model, the algorithm breaks down for wide lenses, such as lenses with FOV greater than 180°, meanwhile the generic model still applies for such lenses.
There is also the comparison between the OpenCV fisheye calibration model to 23-parameter model. 
The calibration pipeline for 23-parameter model:
- Detect and refine the circles from raw images and put them into order. Samples are shown in target_detection folder.
- Initialize the internal parameters using threshold and contour extraction
- Initialize camera pose using back-projection and homography estimation, which still valid for the angle greater than 90° (the p23_model_old.py )
  + The p23_model_old.py uses cv2.solvePnP to estimate the pose. However, it is not generally used for lens that is greater then 180°. In the code, it works because the points detected from the images are within the 180 degree region.
  + The p23_model.py is implemented similar to the paper of Kannala & Brandt's generic camera model and effectively works with high FOV.
- Optimize all the parameters

Usage:
- Circle detection and ordering (similar to run_detection_colab.ipynb)
- Run the calibration (run_calibration.py in p23calibration)

References:
J. Kannala and S. S. Brandt, "A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses," IEEE TPAMI, 2006.
