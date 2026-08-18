import numpy as np
import cv2
from scipy.optimize import least_squares
import fitsio
import matplotlib.pyplot as plt

def fisheye23project(objpoints, rvecs, tvecs, K, D, dt):
  objpoints = np.asarray(objpoints).reshape(-1, 3)
  fx = K[0, 0]
  fy = K[1, 1]
  cx = K[0, 2]
  cy = K[1, 2]
  R, _ = cv2.Rodrigues(rvecs)
  phi = (R @ objpoints.T).T + tvecs
  x = phi[:, 0]/phi[:, 2]
  y = phi[:, 1]/phi[:, 2]
  r = np.sqrt(x**2 + y**2)
  r_safe = np.where(r > 1e-12, r, 1.0)
  theta = np.arctan(r)
  cosp = np.where(r > 1e-12, x / r_safe, 0.0)
  sinp = np.where(r > 1e-12, y / r_safe, 0.0)
  cos2p = 2 * cosp**2 - 1
  sin2p = 2 * sinp * cosp

  D_full = np.concatenate(([1.0], np.ravel(D)))

  rtheta = np.zeros_like(theta)
  thetapow = theta.copy()
  thetasq = theta**2
  for i in D_full:
    rtheta = rtheta + i * thetapow
    thetapow = thetapow * thetasq

  l1, l2, l3, i1, i2, i3, i4, m1, m2, m3, j1, j2, j3, j4 = dt

  deltar = (l1*theta + l2*theta**3 + l3*theta**5)*(i1*cosp + i2*sinp + i3*cos2p + i4*sin2p) #(8)
  deltat = (m1*theta + m2*theta**3 + m3*theta**5)*(j1*cosp + j2*sinp + j3*cos2p + j4*sin2p) #(9)
  dcxd = (rtheta + deltar)*cosp - deltat*sinp #(10)
  dcyd = (rtheta + deltar)*sinp + deltat*cosp #(10)

  u = fx*dcxd + cx #(11)
  v = fy*dcyd + cy #(11)
  return np.column_stack((u, v))

def pack(fx, fy, cx, cy, D, dt, rvecs, tvecs):
  intr = np.hstack([fx, fy, cx, cy, np.ravel(D), np.ravel(dt)])
  extr = np.concatenate([np.concatenate([r, t]) for r, t in zip(rvecs, tvecs)])
  return np.concatenate([intr, extr])

def unpack(p, numimg):
  n = 22
  fx, fy, cx, cy = p[:4]
  D = p[4:8]
  dt = p[8:22]
  extr = p[n:].reshape(numimg, 6)
  rvecs = [extr[i, :3] for i in range(numimg)]
  tvecs = [extr[i, 3:] for i in range(numimg)]
  return fx, fy, cx, cy, D, dt, rvecs, tvecs

def residual(p, objpoints, imgpoints):
  numimg = len(objpoints)
  fx, fy, cx, cy, D, dt, rvecs, tvecs = unpack(p, numimg)
  re = []
  for i in range(numimg):
    proj = fisheye23project(objpoints[i], rvecs[i], tvecs[i], np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), D, dt)
    re.append((imgpoints[i] - proj).ravel())
  return np.concatenate(re)

def fisheye23(objpoints, imgpoints, K, D, dt, rvecs, tvecs):
  numimg = len(imgpoints)
  w, h = (2840, 2840)
  fx0, fy0, cx0, cy0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  K0 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]])

  D0 = np.zeros(4) if D is None else np.array(D, dtype=float).ravel()[:4]

  dt0 = np.zeros(14)
  dt0[0] = 1e-4
  dt0[7] = 1e-4

  rvecs0, tvecs0 = [], []
  for i in range(numimg):
    objpp = objpoints[i].reshape(-1, 1, 3).astype(np.float64)
    imgpp = imgpoints[i].reshape(-1, 1, 2).astype(np.float64)
    ret, rv, tv = cv2.solvePnP(objpp, imgpp, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    rvecs0.append(rv.ravel())
    tvecs0.append(tv.ravel())

  objp = [o.reshape(-1, 3).astype(np.float64) for o in objpoints]
  imgp = [p.reshape(-1, 2).astype(np.float64) for p in imgpoints]
  x0 = pack(fx0, fy0, cx0, cy0, D0, dt0, rvecs0, tvecs0)

  n = 8
  nextr = numimg*6
  lowb = [-np.inf]*n
  highb = [np.inf]*n
  lowb += [-0.05]*14
  highb += [0.05]*14
  lowb += [-np.inf]*nextr
  highb += [np.inf]*nextr
  cali = least_squares(residual, x0, args=(objp, imgp), bounds=(lowb, highb), method='trf', max_nfev=4000, ftol=1e-12, xtol=1e-12)
  fx, fy, cx, cy, D, dt, rvecs, tvecs = unpack(cali.x, numimg)
  K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
  resi = residual(cali.x, objp, imgp).ravel()
  total_points = len(resi) / 2
  rms = np.sqrt(np.sum(resi**2) / total_points)
  return rms, K, D, dt, rvecs, tvecs

def undistort(imgpath, K, D, dt, theta, halfsize):
    size = (2*halfsize, 2*halfsize)
    thetarad = np.radians(theta)
    newf = halfsize/np.tan(thetarad)
    newK = np.array([[newf, 0, halfsize], [0, newK, halfsize],[0, 0, 1]],dtype=np.float64)
    w, h = size
    fx, fy = newK[0,0], newK[1,1]
    cx, cy = newK[0,2], newK[1,2]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    pts3d = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1)
    rvec0 = np.zeros(3)
    tvec0 = np.zeros(3)
    proj = fisheye23project(pts3d, rvec0, tvec0, K, D, dt)
    mapx = proj[:, 0].reshape(h, w).astype(np.float32)
    mapy = proj[:, 1].reshape(h, w).astype(np.float32)

    img = fitsio.read(imgpath)

    undistorted = cv2.remap(img.astype(np.float32), mapx, mapy,interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted 
    
