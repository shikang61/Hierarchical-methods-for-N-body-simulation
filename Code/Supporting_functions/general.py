from Class import Particle
import numpy as np

def reset_particle(particles):
    for p in particles:
        p.phi = 0

def generate_particles(N, size, seed, distribution="random"):
    """
    Generate n particles randomly distributed in a square of size 'size'
    """
    if distribution == "random":
        np.random.seed(seed)
        return [Particle((np.random.uniform(-1*size/2, 1*size/2), np.random.uniform(-1*size/2, 1*size/2)), 1, i+1) for i in range(N)]
    elif distribution == "triangular":
        # generate a concentrated but random distribution of particles
        return [Particle((np.random.triangular(-1*size/4, 0, 1*size/4), np.random.triangular(-1*size/4, 0, 1*size/4)), 1, i+1) for i in range(N)]

def search(box, particle):
    if (len(box.children)==0) and (particle in box.particles): # base case
        return box
    for child in box.children:
        if particle in child.particles:
            box = search(child, particle)
    return box
