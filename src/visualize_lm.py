# See https://github.com/jswanglp/Levenberg-Marquardt-algorithm/blob/master/codes/LM_algorithm.py for reference
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from utils import rosenbrock, rosenbrock_grad

class Strategy:
    def __init__(self, tol=1e-5):
        self.tol = tol

    def rosenbrock(self, x, y, a=1., b=100.):
        return (a - x)**2 + b * (y - x**2)**2

    def draw_chart(self, path, ax, x_lim=[-3., 3], y_lim=[-3, 3], d_off=-1.5):
        x, y = np.meshgrid(np.arange(x_lim[0], x_lim[1], 0.1),
                            np.arange(y_lim[0], y_lim[1], 0.1))
        z = self.rosenbrock(x, y)
        ax.plot_surface(x, y, z, rstride=2, cstride=2, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        # ax.set_zlim([-800.0,2500.0])
        ax.view_init(elev=27, azim=65)
        # z_labels = rosenbrock(np.array(path[0]), np.array(path[1]))
        z_path = self.rosenbrock(np.array(path[0]), np.array(path[1]))
        if path is not None:
            ax.plot(path[0], path[1], z_path, c="#b22222", linewidth=1.)
            ax.scatter(path[0][0], path[1][0], z_path[0], c='r', s=30, marker='o')
            ax.scatter(path[0][-1], path[1][-1], z_path[-1], c='r', s=30, marker='*')
        ax.set_xlim(x_lim), ax.set_ylim(y_lim)
        ax.set_xticks(np.linspace(-2., 2., 5, endpoint=True))
        ax.set_yticks(np.linspace(-2., 2., 5, endpoint=True))
        ax.tick_params(labelsize=14)
    
    def LM_algor(self, x0:np.ndarray, t:int):
        c1, c2 = 4, 0.25
        p0, p1, p2, p3 = 1e-4, 0.5, 0.25, 0.75
        mu_min = 1e-5
        N = x0.shape[0]
        path_x, path_y = [x0[0, 0]], [x0[1, 0]]
        # assert False, path_x

        # Init
        x = x0.copy()
        F, G = rosenbrock(x), rosenbrock_grad(x)
        mu = 1e-2
        l = mu * abs(F)**2 # lambda
        k, s, i, k_ids = 1, 1, 1, [1]
        H = G @ G.T + l * np.eye(N)
        g = -F * G
        while(np.linalg.norm(g) >= self.tol):
            compute_J_flag = False
            dx = np.linalg.solve(H, g)
            r = (abs(F)**2 - rosenbrock(x+dx)**2) / (abs(F)**2 - (F + np.sum(G * dx))**2)
            if r >= p0: x += dx
            F = rosenbrock(x)
            path_x.append(x[0, 0])
            path_y.append(x[1, 0])
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
        return {'final_pos': x, 'iters': k, 'final_grad': G, 'path': [path_x, path_y]}

if __name__ == '__main__':
    
    init_position = np.array([-1.5, -1.5], dtype=np.float64).reshape((2, 1))
    t_ = [5, 10]

    for i, t in zip(range(1, 3), t_):
        fig = plt.figure(i, figsize=(18, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        s = Strategy(tol=1e-5)
        result = s.LM_algor(init_position, t)
        s.draw_chart(result['path'], ax1)
        k = result["iters"]
        ax1.set_title(f"{k} iterations in total with t={t}.")
        x_loc, y_loc = result['final_pos']
        # assert False, x_loc
        print('    Location of the final point: \n    x={:.4f}, y={:.4f}'.format(x_loc[0], y_loc[0]))
        print(result["iters"])

        ax2 = fig.add_subplot(122)
        x_path, y_path = result['path']
        ax2.plot(x_path, y_path, 'r')
        for i in range(0, len(x_path), 1):
            ax2.scatter(x_path[i], y_path[i], c='r', s=30, marker='o')
        ax2.set_xlabel('X', fontsize=14), ax2.set_ylabel('Y', fontsize=14)
        ax2.set_title("Steps taken on X and Y")
        ax2.tick_params(labelsize=14)
        ax2.grid()
    
    plt.show()
