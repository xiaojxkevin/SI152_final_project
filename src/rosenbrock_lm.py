import numpy as np
from utils import rosenbrock, rosenbrock_grad
import time

def original_lm(x0:np.ndarray, tol:float):
    N = x0.shape[0]
    x = x0.copy()
    l, nu = 1e-2, 2
    k = 1
    while True:
        f = rosenbrock(x)
        grad_f = rosenbrock_grad(x)
        H = grad_f @ grad_f.T + l * np.eye(N)
        g = -f * grad_f
        if np.linalg.norm(g) < tol:
            break
        dx = np.linalg.solve(H, g)
        x_new = x + dx
        f_new = rosenbrock(x_new)
        if np.linalg.norm(f_new) < np.linalg.norm(f):
            x = x_new
            l /= nu
        else:
            l *= nu
        k += 1
    return x, k


def adaptive_lm(x0:np.ndarray, tol:float, t:int):
    # Set up some constants
    # And we will fix delta to be 2 
    c1, c2 = 4, 0.25
    p0, p1, p2, p3 = 1e-4, 0.5, 0.25, 0.75
    mu_min = 1e-5
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
        k += 1
        H = G @ G.T + l * np.eye(N)
        g = -F * G
        if compute_J_flag:
            s, i = 1, i + 1
            k_ids.append(k)
        else: s += 1

    return x, k, i

def one(N:int):
    np.random.seed(3)
    x = np.random.randn(N, 1).astype(np.float64)
    # print("Initial x would be\n", x.T, "\n")
    tol = 1e-5

    # hand-writing LM
    # print("-------------Hand Writing LM-------------")
    start = time.time()
    opt_x, numIter = original_lm(x, tol)
    end = time.time() 
    duration = end - start
    hand_result = [numIter, duration, rosenbrock(opt_x)]

    # adaptive multi-step LM
    # print("-------------Adaptive Mulit-step LM-------------")
    adap_result = []
    t_ = [3, 5, 8, 10]
    for t in t_:
        start = time.time()
        opt_x, numIter, i = adaptive_lm(x, tol, t)
        end = time.time()
        duration = end - start
        adap_result.append([numIter, i, duration, rosenbrock(opt_x)])
        # print(opt_x.T)

    return hand_result, adap_result

def main():
    hand_results = [] # Number of Iterations; time; converge; final value
    adap_results = [] # Number of Iterations; steps for J; time; converge; final value
    for n in [2, 8, 20]:
        hand_result, adap_result = one(n)
        hand_results.append(hand_result)
        adap_results.append(adap_result)
        print(f"n: {n} has finished")
    
    np.save("./hand_rosen.npy", np.asarray(hand_results))
    np.save("./adap_rosen.npy", np.asarray(adap_results))

if __name__ == "__main__":
    main()