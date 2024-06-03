import numpy as np

def rosenbrock(x:np.ndarray) -> float:
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

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