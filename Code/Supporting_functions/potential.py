import numpy as np

def potential(target, source):
    """
    Calculate the potential at the target due to the source, using the kernel function:
    phi = qlog(|r|) for 2D problem

    Parameters:
    ---------
    source: Box
        The source box, which contains information about the total mass and centre of mass of a group of particles within it.

    target: Particle
        The target particle, which has information about the charge.
    """
    phi = 0
    phi = target.mass * np.log(distance(target, source.centre_of_mass))
    return phi


def distance(target, source_pos):
    """
    Calculate the distance between the target and the source.

    Parameters:
    ---------
    target: Particle
        The target particle, which has information about the charge.

    source_pos: np.array
        The position of the source.
    """
    r = np.linalg.norm(np.array(target.pos) - np.array(source_pos))
    return r