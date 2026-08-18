import pickle
import numpy as np
from p23_model import fisheye23

spacing = 4.0
rows = 21
cols = 30

loaded = np.load("result/alldata.npz", allow_pickle=True)
grid_matrixes = loaded["grid_matrixes"]
centers_list = loaded["centers"]

alldata = [{"grid_matrix": g, "centers": c} for g, c in zip(grid_matrixes, centers_list)]

imgpoints = []
objpoints = []

for idx, data in enumerate(alldata):
    grid_matrix = data["grid_matrix"]
    centers = np.array(data["centers"], dtype=np.float32)[:, :2]

    current_img = []
    current_obj = []

    crows, ccols = grid_matrix.shape
    for r in range(crows):
        for c in range(ccols):
            pt_idx = grid_matrix[r, c]
            if pt_idx != -1:
                real_r = crows - 1 - r
                real_c = c

                pixel_coords = centers[pt_idx]
                current_img.append(pixel_coords)
                real_x = real_c * spacing
                real_y = real_r * spacing
                current_obj.append([real_x, real_y, 0.0])

    if len(current_img) > 0:
        imgpoints.append(np.array(current_img, dtype=np.float32).reshape(-1, 1, 2))
        objpoints.append(np.array(current_obj, dtype=np.float32).reshape(-1, 1, 3))

w, h = (2840,2840)
K_init = np.matrix([[680.0, 0.0, 1410.0],[0.0, 680.0, 1460.0],[0.0, 0.0, 1.0]],dtype=np.float64)
D_init = np.matrix(np.zeros((5, 1), dtype=np.float64))
dt, tvecs, rvecs = [], [], []
rms, K, D, dt, rvecs, tvecs = fisheye23(objpoints, imgpoints, K_init, D_init, dt, rvecs, tvecs)

print("error: ", rms)
print("K: ", K)
print("D: ", D)
print("dt: ", dt)
print("tvecs: ", tvecs)
print("rvecs: ", rvecs)