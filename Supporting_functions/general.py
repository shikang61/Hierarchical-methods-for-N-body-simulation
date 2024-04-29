from Class import Particle
import numpy as np
from Barnes_Hut_Algo import quotient

def reset_particle(particles):
    """
    This function will reset the potential of all particles to zero

    Inputs:
    -------
    particles : list
        List of particles
    """
    for p in particles:
        p.phi = 0

def generate_particles(N:int, size, seed, distribution="random"):
    """
    This function will generate N number of particles with a given distribution in a square of given size

    Inputs:
    -------
    N : int
        Number of particles
    size : float
        Size of the square
    seed : int
        Seed for random number generator
    distribution : str
        Distribution of the particles: "random" or "triangular"

    Returns:
    --------
    particles : list
        List of particles
    """
    np.random.seed(seed)
    if distribution == "random":
        return [Particle((np.random.uniform(-1*size/2, 1*size/2), np.random.uniform(-1*size/2, 1*size/2)), 1, i+1) for i in range(N)]
    elif distribution == "triangular":
        # generate a concentrated but random distribution of particles
        return [Particle((np.random.triangular(-1*size/4, 0, 1*size/4), np.random.triangular(-1*size/4, 0, 1*size/4)), 1, i+1) for i in range(N)]

def search(box, particle):
    """
    This function returns the leaf box that contains the requested particle

    Returns:
    --------
    box : Box
        The leaf box that contains the requested particle
    """
    if (len(box.children)==0) and (particle in box.particles): # base case
        return box
    for child in box.children:
        if particle in child.particles:
            box = search(child, particle)
    return box

def get_potential(particles):
    """
    This function returns the real part of potential of all particles, then resets the potential to zero for all the particles.

    Returns:
    --------
    phi : np.array
        The potential of all particles
    """
    phi =  np.array([p.phi.real for p in particles])
    reset_particle(particles)
    return phi

def relative_error(algorithm_phi, direct_phi):
    """
    This function calculates the relative error between the potential calculated by the algorithm and the direct sum using the formula:
    error = |direct_phi - algorithm_phi| / |direct_phi|

    Returns:
    --------
    error : np.array
        The relative absolute error
    """
    return np.sqrt(np.sum(np.abs(direct_phi - algorithm_phi)**2) / np.sum(np.abs(direct_phi)**2))

def simple_error(algorithm_phi, direct_phi):
    """
    This function calculates the relative error between the potential calculated by the algorithm and the direct sum using the formula:
    error = |direct_phi - algorithm_phi| / |direct_phi|

    Returns:
    --------
    error : np.array
        The relative absolute error
    """
    return np.abs(direct_phi - algorithm_phi) / np.abs(direct_phi)

def interaction_line(ax, particle, box, max_mass):
    """
    This connects, using a black dotted line, the particle to the mass it is interacting with
    """
    if not box.children: 
        for p in box.particles:
            if p != particle:
                ax.plot([particle.pos[0],p.pos[0]], [particle.pos[1],p.pos[1]], 'k:', alpha=0.3, linewidth=0.9)
    elif quotient(particle, box) < 0.5: 
        ax.plot([particle.pos[0],box.pos[0]], [particle.pos[1], box.pos[1]], 'k:', alpha=0.3, linewidth=0.9)
        ax.scatter(box.pos[0], box.pos[1], s=box.mass/max_mass * 400, facecolors='none', edgecolors='blue')
    else:
        for child in box.children:
                interaction_line(ax, particle, child, max_mass)
    