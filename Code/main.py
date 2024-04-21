from Class import *
from Simulations import *
import numpy as np

# Simulation Parameter:
theta = 0.5
N = 100
size = 16
max_n = 2
np.random.seed(21)
# # 24, 32, 36

particles = [Particle((np.random.uniform(-1*size/2, 1*size/2), np.random.uniform(-1*size/2, 1*size/2)), 1) for _ in range(N)]

def main():
    root = Box_fmm((0, 0), size, c_index=0, parent=None, max_n=max_n)
    plot_tree(root, particles, 0.5, type="fmm", animated=False, get_potential=True)
    
print("start")
main()  
