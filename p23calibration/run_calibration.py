import cv2
import numpy as np
import matplotlib.pyplot as plt
import p23_model as p23
import fitsio

img = fitsio.read("image/image_20260817_10_00_58_00.fits")
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img_gray = np.uint8(img_norm)

fnom = 680.0  
theta_max = np.deg2rad(215.0 / 2.0)

initt = p23.initialize(img_gray, fnom, theta_max)
print(initt)

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
        imgpoints.append(np.array(current_img_pts, dtype=np.float32).reshape(-1, 1, 2))
        objpoints.append(np.array(current_obj_pts, dtype=np.float32).reshape(-1, 1, 3))

K_init = np.array([[initt["fx0"], 0.0, initt["u0"]], [0.0, initt["fy0"], initt["v0"]], [0.0, 0.0, 1.0]],dtype=np.float64,)

H_list = [p23.homo(imgpoints[i], objpoints[i], initt) for i in range(len(imgpoints))]
rms, K, D, dt, rvecs, tvecs = p23.fisheye23(objpoints=objpoints,imgpoints=imgpoints,K=K_init,D=None,dt=None,Ho=H_list,)

print("error: ", rms)
print("K: ", K)
print("D: ", D)
print("dt: ", dt)
print("tvecs: ", tvecs)
print("rvecs: ", rvecs)

np.savez_compressed("result/calibration_23p.npz",rms=rms,K=K,D=D, dt=dt,objpoints=np.array(objpoints, dtype=object),imgpoints=np.array(imgpoints, dtype=object),rvecs=np.array(rvecs, dtype=object),tvecs=np.array(tvecs, dtype=object))