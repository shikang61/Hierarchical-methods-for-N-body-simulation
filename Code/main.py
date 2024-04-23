from Class import *
from Simulations import *
from Supporting_functions.general import generate_particles
import numpy as np


################## "Default Simulation Parameter ##################
theta = 0.5             # for Barnes-Hut
center = (0, 0)         # Center of the simulation area
N = 10                 # Number of particles
size = 128              # Size of the simulation area
max_n = 1               # Maximum number of particles in a leaf box
seed = 21               # Seed for random number generator
distribution = "triangular" # Distribution of the particles: "random" or "triangular"
particles = generate_particles(N, size, seed, distribution)
p = 7                   # Accuracy of the simulation for FMM
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition
####################################################################

def main():
    # root = Box_fmm(center, size, p, max_n=max_n)
    root = Box_bh(center, size, max_n=max_n)
    plot_tree(root, particles, theta, type="bh", animated=False, get_potential=True)

print("start")

main()  
