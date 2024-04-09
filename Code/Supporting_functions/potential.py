import numpy as np

def potential(target, source):
    """
    Calculate the potential at the target due to the source, using the kernel function:
    G = log(|r|) for 2D problem
    ϕ = G * q
    """
    return source.mass * np.log(distance(target, source))

def distance(target, source):
    """
    Calculate the distance between the target particle and the source particle/box.
    """
    return np.linalg.norm(np.array(source.pos) - np.array(target.pos))