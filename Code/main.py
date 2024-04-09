from Class import *
from Barnes_Hut_Algo import *
from Supporting_functions import *

particles = [Particle((np.random.uniform(-1, 1), np.random.uniform(-1, 1)), 1) for _ in range(200)]

def main():
    plot_tree(particles, animated=True)

main()
