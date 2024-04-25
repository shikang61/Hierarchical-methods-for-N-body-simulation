import numpy as np

class Particle():
    """
    The particle class represent the individual objects of interests in the simulation

    Initialisation parameters:
    ---------
    pos: array
        The position of the particle
    mass: float
        The mass of the particle.

    Other attributes:
    ---------
    phi: float
        The potential experienced by the particle due to all other particles in the system.
    idx: int
        The index of the particle in the list of particles.
    """
    def __init__(self, pos, mass, idx):
        self.pos = np.array(pos)
        self.mass= mass
        self.phi = 0
        self.idx = idx