"""
Barnes Hut Algorithm
"""
import time
from Barnes_Hut_Algo.bh_support import BH_potential, quotient

################## QUADTREE BUILDING ##################
def BH_insert_particle(box, particle):
    """
    BH_potential_direct_sum inserts a given particle into a node (box) of the Barnes-Hut Quadtree.
    """
    # update the centre of mass of the box with every insertion of a new particle
    box.update_Centre_Of_Mass(particle) 

    # if the box has vacancy, insert the particle into the box.
    if len(box.particles) < box.max_n: 
        box.particles.append(particle)

     # If no vacancy and the box has no children, create 4 children and redistribute the particle into the appropriate child box.
    elif not box.children:
        box.children = box.create_Children_Boxes()
        particles = box.particles + [particle]
        child_boxes = [box.get_Child_Box(p) for p in particles]
        for p, child_box in zip(particles, child_boxes):
            BH_insert_particle(child_box, p)

    # If no vacancy and the box has children, insert the particle into the appropriate child box.
    else: 
        child_box = box.get_Child_Box(particle)
        BH_insert_particle(child_box, particle)

    if particle not in box.particles:
        box.particles.append(particle)

def BH_build_tree(root, particles):
    """
    Given the root node, this function builds the Barnes-Hut Quadtree from a list of particles given. 
    It Insert the particle one by one to build the tree.
    This algorithm creates an adaptive quadtree.

    Returns:
    --------
    time_taken : float
        Time taken to build the tree
    """
    start_time = time.perf_counter()
    for particle in particles:
        BH_insert_particle(root, particle)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return time_taken

################## CALCULATE POTENTIAL ##################
def BH_calculate_potential_single(particle, box):
    """
    Given a target particle, this function calculates the potential at the 
    particle due to all other particles, using the Barnes-Hut approximation.
    """
    # At the leaf nodes of the Quadtree, compute the potential directly and add to target particle
    if not box.children: 
        for p in box.particles:
            if p != particle:
                particle.phi += BH_potential(particle, p)

    # If the box is far enough, approximate the potential and add to target particle       
    elif quotient(particle, box) < box.theta: 
        particle.phi += BH_potential(particle, box)
    else:
        for child in box.children:
            BH_calculate_potential_single(particle, child)

def BH_calculate_potential_all(particles, root):
    """
    This function computes the potential for every particle in the Quadtree.

    Returns:
    --------
    time_taken : float
        Time taken to calculate the potential for all particles
    """
    start_time = time.perf_counter()
    for p in particles:
        BH_calculate_potential_single(p, root)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return time_taken


    



