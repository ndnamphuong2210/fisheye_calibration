import numpy as np
import cv2
from scipy.optimize import least_squares
import fitsio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    
def plotallresidual(imgpoints, objpoints, rvecs, tvecs, K, D, dt, size = (2840, 2840), maxrad = 1000, bin = 5 ):
    dx, dy, err, radius = [], [], [], []
    x, y = [], []
    w, h = size
    cx = K[0, 2]
    cy = K[1, 2]
    for i in range(len(imgpoints)):
        reproj = fisheye23project(objpoints[i].astype(np.float32), rvecs[i].astype(np.float32), tvecs[i].astype(np.float32), K, D, dt)
        reproj = reproj.reshape(-1, 2)
        dots = imgpoints[i].reshape(-1, 2).astype(np.float32)

        xa = dots[:, 0]
        ya = dots[:, 1]
        deltax = xa - reproj[:, 0]
        deltay = ya - reproj[:, 1]
        error = np.sqrt(deltax**2 + deltay**2)
        rad = np.sqrt((xa - cx) ** 2 + (ya - cy) ** 2)

        dx.append(deltax)
        dy.append(deltay)
        err.append(error)
        radius.append(rad)
        x.append(xa)
        y.append(ya)

    alldx = np.concatenate(dx)
    alldy = np.concatenate(dy)
    allerr = np.concatenate(err)
    allradius = np.concatenate(radius)
    allx = np.concatenate(x)
    ally = np.concatenate(y)

    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.scatter(allx, alldx, alpha=0.4, c='blue', edgecolors='none', s=12)
    plt.xlim(0, w)
    plt.ylim(-4, 4)
    plt.title("error", fontsize=15)
    plt.xlabel("x", fontsize=13)
    plt.ylabel("dx", fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(222)
    plt.scatter(ally, alldy, alpha=0.4, c='blue', edgecolors='none', s=12)
    plt.xlim(0, h)
    plt.ylim(-4, 4)
    plt.title("error", fontsize=15)
    plt.xlabel("y", fontsize=13)
    plt.ylabel("dy", fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(223)
    plt.scatter(allradius, allerr, alpha=0.4, c='blue', edgecolors='none', s=12)
    plt.xlim(0, 1000)
    plt.ylim(0, 4)
    plt.title("rms error", fontsize=15)
    plt.xlabel("radius", fontsize=13)
    plt.ylabel("residual", fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)

    bin = bin
    r = maxrad
    binedges = np.arange(0, r + bin, bin)
    bin_centers = binedges[:-1] + bin / 2
    bin_rms = np.full(len(bin_centers), np.nan)
    bin_count = np.zeros(len(bin_centers), dtype=int)

    bin_idx = np.digitize(allradius, binedges) - 1
    for b in range(len(bin_centers)):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_rms[b] = np.sqrt(np.mean(allerr[mask] ** 2))
            bin_count[b] = mask.sum()
    plt.subplot(224)
    valid = ~np.isnan(bin_rms)
    plt.plot(bin_centers[valid], bin_rms[valid], marker='o', markersize=4, linewidth=1.5)
    plt.xlim(0, r)
    plt.ylim(0, 2)
    plt.title("RMS error", fontsize=15)
    plt.xlabel("radius", fontsize=13)
    plt.ylabel("residual", fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

def ploteach(imgpoints, objpoints, rvecs, tvecs, K, D, dt, size = (2840, 2840), maxrad = 1000, bin = 5):
    dx, dy, err, radius = [], [], [], []
    x, y = [], []
    w, h = size
    cx = K[0, 2]
    cy = K[1, 2]
    for i in range(len(imgpoints)):
        reproj = fisheye23project(objpoints[i].astype(np.float32), rvecs[i].astype(np.float32), tvecs[i].astype(np.float32), K, D, dt)
        reproj = reproj.reshape(-1, 2)
        dots = imgpoints[i].reshape(-1, 2).astype(np.float32)
   
        xa = dots[:, 0]
        ya = dots[:, 1]
        deltax = xa - reproj[:, 0]
        deltay = ya - reproj[:, 1]
        error = np.sqrt(deltax**2 + deltay**2)
        rad = np.sqrt((xa - cx) ** 2 + (ya - cy) ** 2)
   
        dx.append(deltax)
        dy.append(deltay)
        err.append(error)
        radius.append(rad)
        x.append(xa)
        y.append(ya)

        plt.figure(figsize=(12, 11))

        plt.subplot(221)
        plt.scatter(rad, error, alpha=0.4, c="blue", edgecolors="none", s=12)
        plt.xlim(0, maxrad)
        plt.ylim(0, 4)
        plt.title(f"Residual vs radius (image {i+1})", fontsize=12)
        plt.xlabel("Radius (px)", fontsize=10)
        plt.ylabel("Residual (px)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)

        plt.subplot(222)
        ax2 = plt.gca()
        q_plot = ax2.quiver( xa, ya, deltax, deltay, error, cmap="jet", angles="xy", scale_units="xy", scale=0.01, width=0.003, clim=(0, 5))

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(q_plot, cax=cax)

        margin = 50
        ax2.set_xlim(xa.min() - margin, xa.max() + margin)
        ax2.set_ylim(ya.max() + margin, ya.min() - margin)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_title(f"Residual vector (image {i+1})", fontsize=12)
        ax2.set_xlabel("x (px)", fontsize=10)
        ax2.set_ylabel("y (px)", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.subplot(223)
        plt.scatter(xa, deltax, alpha=0.4, c="blue", edgecolors="none", s=12)
        plt.xlim(0, size[0])
        plt.ylim(-4, 4)
        plt.title(f"dx vs X (image {i+1})", fontsize=12)
        plt.xlabel("x (px)", fontsize=10)
        plt.ylabel("dx (px)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)

        plt.subplot(224)
        plt.scatter(ya, deltay, alpha=0.4, c="blue", edgecolors="none", s=12)
        plt.xlim(0, size[1])
        plt.ylim(-4, 4)
        plt.title(f"dy vs Y (image {i+1})", fontsize=12)
        plt.xlabel("y (px)", fontsize=10)
        plt.ylabel("dy (px)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)