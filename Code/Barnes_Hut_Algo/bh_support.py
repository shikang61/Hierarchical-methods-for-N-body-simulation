import time
import numpy as np

################## POTENTIAL ################
def BH_potential(target, source):
    """
    Calculate the potential at the target due to the source, using the kernel function:
    G = log(|r|) for 2D problem
    ϕ = q * G

    Returns:
    --------
    potential : float
        The potential at the target due to the source
    """
    return source.mass * np.log(distance(target, source))

def distance(target, source):
    """
    Calculate the distance between the target particle and the source particle/box.
    
    Returns:
    -------
    distance : float
        The distance between the target and source
    """
    return np.linalg.norm(np.array(source.pos) - np.array(target.pos))

def quotient(particle, box):
    """
    This function computes the theta parameter for the Barnes-Hut approximation.

    Returns:
    --------
    quotient : float
        The ratio of the size of the box to the distance between the particle and the box
    """
    return box.size / distance(particle, box)

def BH_potential_direct_sum(particles):
    """
    Direct sum calculation of all-to-all potential from seperate sources

    Returns:
    --------
    time_taken : float
        Time taken to calculate the potential directly
    """
    start_time = time.perf_counter()
    for i, particle in enumerate(particles):
        for source in (particles[:i] + particles[i+1:]):
            particle.phi += BH_potential(particle, source)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return time_taken