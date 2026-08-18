import pickle
import numpy as np
import cv2 #opencv-python==4.10.0.84

spacing = 4.0
rows = 21
cols = 30

npz_path = "result/alldataa.npz" 
data = np.load(npz_path, allow_pickle=True)

grids_list = data["grids"]
centers_list = data["centers"]

imgpoints = []
objpoints = []

for grid_matrix, centers in zip(grids_list, centers_list):
    grid_matrix = np.array(grid_matrix)
    centers = np.array(centers, dtype=np.float32)[:, :2]

    current_img_pts = []
    current_obj_pts = []

    curr_rows, curr_cols = grid_matrix.shape
    for r in range(curr_rows):
        for c in range(curr_cols):
            pt_idx = grid_matrix[r, c]
            if pt_idx != -1:
                real_r = curr_rows - 1 - r
                real_c = c

                pixel_coords = centers[pt_idx]
                current_img_pts.append(pixel_coords)
                real_x = real_c * spacing
                real_y = real_r * spacing
                current_obj_pts.append([real_x, real_y, 0.0])

    if len(current_img_pts) > 0:
        imgpoints.append(
            np.array(current_img_pts, dtype=np.float32).reshape(-1, 1, 2)
        )
        objpoints.append(
            np.array(current_obj_pts, dtype=np.float32).reshape(-1, 1, 3)
        )

h, w = (2840,2840)
K_init = np.matrix([[680.0, 0.0, 1410.0],[0.0, 680.0, 1460.0],[0.0, 0.0, 1.0]],dtype=np.float64)
D_init = np.matrix(np.zeros((4, 1), dtype=np.float64))
flags = (cv2. fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, (w, h), K_init, D_init, flags=flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-10))

print("error: ", rms)
print("K: ", K)
print("D: ", D)
print("tvecs: ", tvecs)
print("rvecs: ", rvecs)

np.savez_compressed("result/calibration_opencv.npz",rms=rms,K=K,D=D,objpoints=np.array(objpoints, dtype=object),imgpoints=np.array(imgpoints, dtype=object),rvecs=np.array(rvecs, dtype=object),tvecs=np.array(tvecs, dtype=object))
