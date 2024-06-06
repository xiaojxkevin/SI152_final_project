import numpy as np
from numpy import cos, sin
from math import pi

def rosenbrock(x:np.ndarray) -> float:
    return float(sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)))

def rosenbrock_grad(x:np.ndarray) -> np.ndarray:
    N = x.shape[0]
    if N <= 1: assert False, f"Wrong Input!"
    grad = np.zeros_like(x, x.dtype)
    grad[0] = 400 * x[0] * (x[0]**2 - x[1]) + 2 * (x[0] - 1)
    grad[N-1] = 200 * (x[N-1] - x[N-2]**2)
    if N == 2: return grad
    for i in range(1, N-1):
        grad[i] = 200 * (x[i] - x[i-1]**2) + 400 * x[i] * (x[i]**2 - x[i+1]) + 2 * (x[i] - 1)
    return grad

def IK_pose(x:np.ndarray, l1:float, l2:float) -> np.ndarray:
    """
    x should be in shape (2, 1)
    """
    assert x.shape == (2, 1)
    theta_1, theta_2 = x[0, 0], x[1, 0]
    assert 0 < theta_1 < pi / 2 and -pi / 2 < theta_2 < pi / 2
    F = np.zeros_like(x, dtype=np.float64)
    F[0, 0] = l2 * cos(theta_1 + theta_2) + l1 * cos(theta_1)
    F[1, 0] = l2 * sin(theta_1 + theta_2) + l1 * sin(theta_1)
    return F

def IK(F:np.ndarray, F_gt:np.ndarray):
    return F - F_gt

def IK_grad(x:np.ndarray, l1:float, l2:float) -> np.ndarray:
    theta_1, theta_2 = x[0, 0], x[1, 0]
    grad = np.zeros((2, 2), dtype=np.float64)
    grad[0, 0] = -(l2 * sin(theta_1 + theta_2) + l1 * sin(theta_1))
    grad[0, 1] = -l1 * sin(theta_1 + theta_2)
    grad[1, 0] = l2 * cos(theta_1 + theta_2) + l1 * cos(theta_1)
    grad[1, 1] = l1 * cos(theta_1 + theta_2)
    return grad
