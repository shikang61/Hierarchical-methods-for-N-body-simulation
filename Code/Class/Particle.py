import numpy as np

class Particle():
    """
    The particle class represent the individual objects of interests in the simulation
    """
    def __init__(self, pos, mass):
        self.pos = np.array(pos)
        self.mass = mass