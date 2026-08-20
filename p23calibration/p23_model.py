import numpy as np
import cv2
from scipy.optimize import least_squares
import fitsio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def initialize(img_gray, fnom, theta_rad):
    desired_projection = lambda th, f: f * th
    thetas = np.linspace(0, theta_rad, 200)
    r_desired = desired_projection(thetas, fnom)
    A = np.column_stack([thetas, thetas**3])
    k1, k2 = np.linalg.lstsq(A, r_desired, rcond=None)[0]
    rmax = k1 * theta_rad + k2 * (theta_rad**3)
    _, lens = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    lens = cv2.medianBlur(lens, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(lens, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lens_contour = max(contours, key=cv2.contourArea)
    ec = cv2.fitEllipse(lens_contour)
    (u0, v0), (dx, dy), angle = ec

    a = dx / 2.0
    b = dy / 2.0

    mu = a / rmax
    mv = b / rmax
    fx0 = k1 * mu
    fy0 = k1 * mv

    return dict( k1=k1, k2=k2, u0=u0, v0=v0, mu=mu, mv=mv, fx0=fx0, fy0=fy0, rmax=rmax)


def homo(imgpoints, objpoints, initt):
    pts = imgpoints.reshape(-1, 2)
    k1, k2 = initt["k1"], initt["k2"]
    u0, v0 = initt["u0"], initt["v0"]
    fx = initt.get("fx0", k1 * initt["mu"])
    fy = initt.get("fy0", k1 * initt["mv"])

    xn = (pts[:, 0] - u0) / fx
    yn = (pts[:, 1] - v0) / fy

    r = np.sqrt(xn**2 + yn**2)
    theta = np.zeros_like(r)

    if np.abs(k2) < 1e-12:
        theta = r / k1
    else:
        p_coef = k1 / k2 
        q_coef = -r / k2
        disc = (q_coef / 2) ** 2 + (p_coef / 3) ** 3

        i_pos = np.where(disc >= 0)[0]
        if len(i_pos) > 0:
            qd = q_coef[i_pos]
            sqrt_d = np.sqrt(disc[i_pos])
            u = np.cbrt(-qd / 2 + sqrt_d)
            v = np.cbrt(-qd / 2 - sqrt_d)
            theta[i_pos] = u + v

        i_neg = np.where(disc < 0)[0]
        if len(i_neg) > 0:
            qd = q_coef[i_neg]
            m = 2 * np.sqrt(-p_coef / 3)
            arg = np.clip(3 * qd / (p_coef * m), -1, 1)
            phi_ang = np.arccos(arg) / 3
            t0 = m * np.cos(phi_ang)
            t1 = m * np.cos(phi_ang - 2 * np.pi / 3)
            t2 = m * np.cos(phi_ang - 4 * np.pi / 3)
            candidates = np.stack([t0, t1, t2], axis=1)
            candidates_pos = np.where(candidates > 0, candidates, np.inf)
            theta[i_neg] = np.min(candidates_pos, axis=1)

    theta = np.clip(theta, 0, np.pi - 1e-6)
    r_safe = np.where(r > 1e-10, r, 1.0)
    sinp = np.where(r > 1e-10, xn / r_safe, 0.0) 
    cosp = np.where(r > 1e-10, yn / r_safe, 0.0)

    sphere = np.column_stack([sinp * np.sin(theta), cosp * np.sin(theta), np.cos(theta)])

    obje = np.asarray(objpoints).reshape(-1, 3)[:, :2] if np.asarray(objpoints).shape[-1] == 3 else np.asarray(objpoints).reshape(-1, 2)
    cen = np.mean(obje, axis=0)
    d = obje - cen
    mean_d = np.mean(np.linalg.norm(d, axis=1))
    scale = np.sqrt(2) / (mean_d if mean_d >= 1e-12 else 1.0)

    T = np.array([[scale, 0, -scale * cen[0]],[0, scale, -scale * cen[1]], [0, 0, 1]])
    n = obje.shape[0]
    obje_h = np.column_stack([obje, np.ones(n)])
    obje_n = (T @ obje_h.T).T[:, :2]
    X = np.column_stack([obje_n, np.ones(n)])

    A = []
    for i in range(n):
        Xi = X[i]
        mx, my, mz = sphere[i]
        zero = np.zeros(3)
        A.append(np.concatenate([zero, -mz * Xi, my * Xi]))
        A.append(np.concatenate([mz * Xi, zero, -mx * Xi]))

    _, _, Vt = np.linalg.svd(np.array(A))
    Hp = Vt[-1].reshape(3, 3)
    H_init = Hp @ T

    X_unnorm = np.column_stack([obje, np.ones(n)])


    def residual_H(h):
        H = h.reshape(3, 3)
        proj = (H @ X_unnorm.T).T
        norm = np.linalg.norm(proj,axis=1,keepdims=True)
        xh = proj / np.maximum(norm, 1e-12)
        cross = np.cross(sphere, xh)
        return cross.ravel()

    res = least_squares(residual_H, H_init.ravel(), method="lm", max_nfev=2000)
    H_refined = res.x.reshape(3, 3)

    return H_refined


def extrinsic_estimation(H):
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    nu1 = np.linalg.norm(h1)
    if nu1 < 1e-12:
        print("nooo homography")
    sign = 1.0 if H[2, 2] >= 0 else -1.0
    lam = sign / nu1
    r1, r2 = h1*lam, h2*lam
    r3 = np.cross(r1, r2)
    t = h3 * lam
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), t.ravel()

def fisheye23project(objpoints, rvecs, tvecs, K, D, dt):
    objpoints = np.asarray(objpoints).reshape(-1, 3)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    R, _ = cv2.Rodrigues(rvecs)
    phi = (R @ objpoints.T).T + tvecs

    x, y, z = phi[:, 0], phi[:, 1], phi[:, 2]  
    r = np.sqrt(x**2 + y**2)
    r_safe = np.where(r > 1e-12, r, 1.0)
    theta = np.arctan2(r, z)                    
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


def fisheye23(objpoints, imgpoints, K, D, dt, Ho):
    numimg = len(imgpoints)
    fx0, fy0, cx0, cy0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    D0 = np.zeros(4) if D is None else np.array(D, dtype=float).ravel()[:4]
    dt0 = np.zeros(14)
    dt0[0] = 1e-4
    dt0[7] = 1e-4

    rvecs0, tvecs0 = [], []
    for H in Ho:
        rv, tv = extrinsic_estimation(H)
        rvecs0.append(rv)
        tvecs0.append(tv)

    objp = [o.reshape(-1, 3).astype(np.float64) for o in objpoints]
    imgp = [p.reshape(-1, 2).astype(np.float64) for p in imgpoints]

    x0 = pack(fx0, fy0, cx0, cy0, D0, dt0, rvecs0, tvecs0)

    lowb = [100.0, 100.0, cx0 - 300.0, cy0 - 300.0] + [-5.0]*4 + [-0.05]*14 + [-np.inf]*(numimg*6)
    highb = [3000.0, 3000.0, cx0 + 300.0, cy0 + 300.0] + [5.0]*4 + [0.05]*14 + [np.inf]*(numimg*6)

    cali = least_squares(residual, x0, args=(objp, imgp), bounds=(lowb, highb),method="trf", x_scale="jac", max_nfev=3000, ftol=1e-8, xtol=1e-8, verbose=2,)

    fx, fy, cx, cy, D, dt, rvecs, tvecs = unpack(cali.x, numimg)
    K_opt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    resi = residual(cali.x, objp, imgp).ravel()
    total_points = len(resi) / 2
    rms = np.sqrt(np.sum(resi**2) / total_points)

    return rms, K_opt, D, dt, rvecs, tvecs



def undistort(imgpath, K, D, dt, theta, halfsize):
    size = (2*halfsize, 2*halfsize)
    thetarad = np.radians(theta)
    newf = halfsize/np.tan(thetarad)
    newK = np.array([[newf, 0, halfsize], [0, newf, halfsize],[0, 0, 1]],dtype=np.float64)
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

    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)
    plt.savefig("result/residualall_p23.png", dpi=300)
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

        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)
        plt.savefig(f"result/residualeach_p23/residual_image_{i+1}.png", dpi=300)
        plt.show()






