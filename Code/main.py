from Class import *
from Simulations import *
import numpy as np

# Simulation Parameter:
theta = 0.5
N = 10
size = 10

particles = [Particle((np.random.uniform(-1*size/2, 1*size/2), np.random.uniform(-1*size/2, 1*size/2)), 1) for _ in range(N)]
root = Box((0, 0), size, None)
def main():
    plot_tree(root, particles, 0.5, animated=False, get_potential=True)
    


main()
