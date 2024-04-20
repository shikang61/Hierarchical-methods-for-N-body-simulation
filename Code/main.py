from Class import *
from Simulations import *
import numpy as np

# Simulation Parameter:
theta = 0.5
N = 100
size = 10

# # 24, 32, 36
# np.random.seed(50)

particles = [Particle((np.random.uniform(-1*size/2, 1*size/2), np.random.uniform(-1*size/2, 1*size/2)), 1) for _ in range(N)]


def main():
    root = Box_fmm((0, 0), size, None)
    plot_tree(root, particles, 0.5, type="fmm", animated=True, get_potential=False)
    
print("start")
main()
