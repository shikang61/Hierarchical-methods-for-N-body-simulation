from Class import *
from Simulations import *
import numpy as np

# Simulation Parameter:
theta = 0.5
N = 10
size = 10

particles = [Particle((np.random.uniform(-1*size/2, 1*size/2), np.random.uniform(-1*size/2, 1*size/2)), 1) for _ in range(N)]


def main():
    root = Box_fmm((0, 0), size, None)
    print(root)
    plot_tree(root, particles, 0.5, type="fmm", animated=False, get_potential=False)
    

print("start")
main()
