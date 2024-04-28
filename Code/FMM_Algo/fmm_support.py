import numpy as np
import math
from tqdm import tqdm
import time
################## QUADTREE ##################
def get_neighbours_child(box, curr_child, i, other):
    """
    Get the child of the parent's neighbours
    """
    parent_neighbours_child = curr_child ^ ((other+1)%2+1)
    if box.side_neighbours[i] is not None:
        return box.side_neighbours[i].get_Child_At_Index(parent_neighbours_child)
    else:
        return None

def get_corner_neighbours(box):
    """
    Gets the corner neighbours of the current box.

    Returns:
    --------
    corner_neighbour : list
        List of corner neighbours of the current box
    """
    corner_neighbour = []
    DISREGARD = [1,2,0,3]
    CORNER_CHILDREN = (3, 2, 0, 1)
    for i, side_neigh in enumerate(box.side_neighbours):
        # Find remaining corner neighours at lower levels
        if side_neigh is not None:
            corner = side_neigh.side_neighbours[(i+1)%4]
            if corner is not None:
                if side_neigh.level == box.level:
                    # Find corner neighbours if side neighbour is same level as current box
                    """
                        | 1 >   | 
                    ^^^^---------
                      0 | * | 2 | 
                    ---------⌄⌄⌄⌄
                        < 3 |   | 
                    NW corner neighbour is side neighbour 1 of current box's neighbour 0
                    NE corner neighbour is side neighbour 2 of current box's neighbour 1
                    SE corner neighbour is side neighbour 3 of current box's neighbour 2
                    SW corner neighbour is side neighbour 0 of current box's neighbour 3
                    """
                    corner_neighbour.append(corner)
                elif side_neigh.level < box.level and i != DISREGARD[box.c_index]:
                    # Find corner neighbours if side neighbour is lower level than current box
                    """
                        ---------
                        |       |  
                        |   1   |  
                        |       | 
                -------------------------
                |       | 0 | 1 |       |
                |   0   ---------   2   |   
                |       | 2 | 3 |       |
                -------------------------
                        |       |   
                        |   3   | 
                        |       | 
                        ---------
                    If cindex == 0, disregard side neighbour 1
                    If cindex == 1, disregard side neighbour 2
                    If cindex == 2, disregard side neighbour 0
                    If cindex == 3, disregard side neighbour 3

                    NW corner neighbour is child 3 of side neighbour 0
                    NE corner neighbour is child 2 of side neighbour 1
                    SE corner neighbour is child 0 of side neighbour 2
                    SW corner neighbour is child 1 of side neighbour 3
                    """
                    corner_neighbour.append(corner.get_Child_At_Index(CORNER_CHILDREN[i]))
    return corner_neighbour

################## POTENTIAL ################## 
def FMM_potential(target, source):
    """
    Calculate the potential at the target due to the source, using the kernel function:
    G = -log(|r|) for 2D problem
    ϕ = q * G

    Returns:
    --------
    potential : float
        The potential at the target due to the source
    """
    return source.mass * np.log(complex(*target.pos) - complex(*source.pos))

def evaluatate_local_potential(box, time_dic): #L2P
    """
    Evaluate the local potential at the particles in the box
    ϕ = a_0 + a_1*(z-z0) + a_2*(z-z0)^2 + ... + a_p*(z-z0)^p

    polyeval: np.polyval([1,2,3], 4) -> 1*4**2 + 2*4**1 + 3*4**0

    Returns:
    --------
    time_dic : dict
        Dictionary of time taken for each operation
    """
    z0 = complex(*box.coords)
    coeffs = box.inner_coeffs
    start_time = time.perf_counter()
    for particle in box.particles: #L2P
        z = complex(*particle.pos)
        particle.phi += np.real(np.polyval(coeffs[::-1], z-z0))
    end_time = time.perf_counter()
    time_dic["L2P_time"] += end_time - start_time
    return time_dic

def potential_direct_sum_nearest_neighbour(particles, sources, time_dic): #P2P_1
    """
    Direct sum calculation of all-to-all potential from seperate sources
    """
    start_time = time.perf_counter()
    for particle in particles:
        for source in sources:
            particle.phi += FMM_potential(particle, source)
    end_time = time.perf_counter()
    time_dic["P2P_time"] += end_time - start_time
    return time_dic
            
def FMM_potential_direct_sum(particles, tqdm_bar=False): #P2P_2
    """
    Direct sum calculation of pairwise potential between all particles given

    Inputs:
    --------
    particles : list
        List of particles to calculate pairwise potential
    tqdm_bar : bool
        If True, display a progress bar

    Returns:
    --------
    time_taken : float
        Time taken to calculate the potential directly
    """
    start_time = time.perf_counter()
    for i, particle in enumerate(tqdm(particles, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable= not tqdm_bar, desc="Direct sum progress: ")):
        for source in (particles[:i] + particles[i+1:]):
            particle.phi += FMM_potential(particle, source)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return time_taken


################## MISCELLANEOUS ##################
def find_p(accuracy):
    """
    Find the number of terms p needed to satisfy the error condition

    Returns:
    --------
    p : int
        Number of terms needed to satisfy the error condition
    """
    return math.ceil(-np.log2(accuracy))

def FMM_search(particle, box):
    """
    Perform a earch for the leaf box containing the particle

    Returns:
    --------
    box : Box_fmm
        The leaf box containing the particle
    """
    if len(box.children) != 0:
        next_box = box.get_Child_Box(particle)
        return FMM_search(particle, next_box)
    else:
        return box
    
def set_max_levels(N, n_leaf):
    """
    Set the maximum levels of the tree based on the number of particles and the maximum number of particles in a leaf box

    Returns:
    --------
    max_levels : int
        Maximum levels of the tree
    """
    return int(np.ceil(np.log(N/n_leaf)/np.log(4)))
