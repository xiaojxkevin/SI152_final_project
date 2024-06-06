import numpy as np
from numpy.linalg import norm
from math import pi
from utils import IK, IK_grad, IK_pose


def main():
    np.random.seed(33)
    tol, t, l1, l2 = 1e-5, 5, 1.0, 0.5
    x_gt = np.array([pi / 3, -pi / 4]).reshape((2, 1))
    F_gt = IK_pose(x_gt, l1, l2)
    print(f"The Initial x: {np.rad2deg(x_gt).T}, coord: {F_gt.T}")
    # x0 = np.array([pi / 10, -pi / 3], dtype=np.float64).reshape((2, 1))
    x0 = np.array([pi / 10, 0], dtype=np.float64).reshape((2, 1))
    opt_x, k, i = adaptive_lm(x0, tol, t, l1, l2, F_gt)
    print(np.rad2deg(opt_x).T)
    print(k, i)

def adaptive_lm(x0:np.ndarray, tol:float, t:int, l1:float, l2:float, F_gt:np.ndarray):
    # Set up some constants
    # And we will fix delta to be 2 
    c1, c2 = 4, 0.25
    p0, p1, p2, p3 = 1e-4, 0.5, 0.25, 0.75
    mu_min = 1e-5
    N = x0.shape[0]

    # Init
    x = x0.copy()
    F, G = IK(IK_pose(x, l1, l2), F_gt), IK_grad(x, l1, l2)
    mu = 1e-2
    l = mu * norm(F)**2 # lambda
    k, s, i, k_ids = 1, 1, 1, [1]
    H = G.T @ G + l * np.eye(N)
    g = -G.T @ F
    while(np.linalg.norm(g) >= tol):
        compute_J_flag = False
        dx = np.linalg.solve(H, g).reshape((2, 1))
        r = (norm(F)**2 - norm(IK(IK_pose(x+dx, l1, l2), F_gt))**2) / (norm(F)**2 - norm(F + G @ dx)**2)
        if r >= p0: x += dx
        F = IK(IK_pose(x, l1, l2), F_gt)
        if r < p2: mu *= c1
        elif r > p3: mu = max(mu_min, c2 * mu)
        if r < p1 or s >= t:
            G = IK_grad(x, l1, l2)
            l = mu * norm(F)**2
            compute_J_flag = True
        k += 1
        H = G.T @ G + l * np.eye(N)
        g = -G.T @ F
        if compute_J_flag:
            s, i = 1, i + 1
            k_ids.append(k)
        else: s += 1

    return x, k, i

if __name__ == "__main__":
    main()