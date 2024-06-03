import numpy as np
from utils import rosenbrock, rosenbrock_grad
import time

def original_lm(x0:np.ndarray, tol:float, maxIter:int):
    N = x0.shape[0]
    x = x0.copy()
    exitFlag = False
    l, nu = 1e-2, 2
    for k in range(1, maxIter + 1):
        f = rosenbrock(x)
        grad_f = rosenbrock_grad(x)
        H = grad_f @ grad_f.T + l * np.eye(N)
        g = -f * grad_f
        if np.linalg.norm(g) < tol:
            exitFlag = True
            break
        dx = np.linalg.solve(H, g)
        x_new = x + dx
        f_new = rosenbrock(x_new)
        if np.linalg.norm(f_new) < np.linalg.norm(f):
            x = x_new
            l /= nu
        else:
            l *= nu
    if not exitFlag: print(f"<<<<<<< Algorithm does not converge! >>>>>>>")
    return x, k


def adaptive_lm(x0:np.ndarray, tol:float, maxIter:int):
    # Set up some constants
    # And we will fix delta to be 2 
    c1, c2 = 4, 0.25
    p0, p1, p2, p3 = 1e-4, 0.5, 0.25, 0.75
    mu_min = 1e-5
    t = 5
    N = x0.shape[0]

    # Init
    x = x0.copy()
    F, G = rosenbrock(x), rosenbrock_grad(x)
    mu = 1e-2
    l = mu * abs(F)**2 # lambda
    k, s, i, k_ids = 1, 1, 1, [1]
    H = G @ G.T + l * np.eye(N)
    g = -F * G
    while(np.linalg.norm(g) >= tol):
        compute_J_flag = False
        dx = np.linalg.solve(H, g)
        r = (abs(F)**2 - rosenbrock(x+dx)**2) / (abs(F)**2 - (F + np.sum(G * dx))**2)
        if r >= p0: x += dx
        F = rosenbrock(x)
        if r < p2: mu *= c1
        elif r > p3: mu = max(mu_min, c2 * mu)
        if r < p1 or s >= t:
            G = rosenbrock_grad(x)
            l = mu * abs(F)**2
            compute_J_flag = True
        if k == maxIter:
            print(f"<<<<<<< Algorithm does not converge! >>>>>>>")
            break
        k += 1
        H = G @ G.T + l * np.eye(N)
        g = -F * G
        if compute_J_flag:
            s, i = 1, i + 1
            k_ids.append(k)
        else: s += 1

    return x, k, i

def main():
    N = 20
    x = np.random.randn(N, 1).astype(np.float64)
    maxIter = 300 * (N + 1)
    # x = np.array([1, 1], dtype=np.float64).reshape((2, 1))
    print("Initial x would be\n", x.T)
    tol = 1e-5

    # hand-writing LM
    print("-------------Hand Writing LM-------------")
    start = time.time()
    opt_x, numIter = original_lm(x, tol, maxIter)
    end = time.time() 
    duration = end - start
    print(opt_x.T)
    print(f"Numbe of iterations: {numIter} and time duration {duration}")
    print(f"The final result: {rosenbrock(opt_x)}")

    print()

    # adaptive multi-step LM
    print("-------------Adaptive Mulit-step LM-------------")
    start = time.time()
    opt_x, numIter, i = adaptive_lm(x, tol, maxIter)
    end = time.time()
    duration = end - start
    print(opt_x.T)
    print(f"Numbe of iterations: {numIter} and time duration {duration}")
    print(f"The final result: {rosenbrock(opt_x)}")

if __name__ == "__main__":
    main()