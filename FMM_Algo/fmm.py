"""
Fast Muiltipole Method (FMM) algorithm
"""

import numpy as np
from .fmm_support import potential_direct_sum_nearest_neighbour, FMM_potential_direct_sum, FMM_search, evaluatate_local_potential
import math
import time

################## QUADTREE BUILDING ##################
### Adaptive quadtree ###
def FMM_insert_particle(box, particle):
    """
    This function inserts a particle into the Barnes-Hut Quadtree.
    """
    if len(box.particles) < box.max_n: # if the box has vacancy, insert the particle into the box.
        box.particles.append(particle)
    elif not box.children: # If no vacancy and the box has no children, create 4 children and redistribute the particle into the appropriate child box.
        box.children = box.create_Children_Boxes()
        particles = box.particles + [particle]
        child_boxes = [box.get_Child_Box(p) for p in particles]
        for p, child_box in zip(particles, child_boxes):
            FMM_insert_particle(child_box, p)
    else: # If no vacancy and the box has children, insert the particle into the appropriate child box.
        child_box = box.get_Child_Box(particle)
        FMM_insert_particle(child_box, particle)
    if particle not in box.particles:
        box.particles.append(particle)

def FMM_build_adaptive_tree(root, particles):
    """
    This function builds the FMM Adaptive Quadtree from a list of particles given.

    Returns:
    --------
    time_taken : float
        Time taken to build the tree
    """
    start_time = time.perf_counter()
    for particle in particles:
        FMM_insert_particle(root, particle)
    root.set_Child_Side_Neighbors()
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return time_taken

### Quadtree with a fixed number of levels ###
def FMM_create_tree(box, levels):
    """
    This function creates the FMM Quadtree with a fixed number of levels.
    """
    if box.level < levels:
        if not box.children:
            box.children = box.create_Children_Boxes()
            for child in box.children:
                FMM_create_tree(child, levels)

def FMM_populate_tree(particles, root):
    """
    This function populates the FMM Quadtree with the particles given.
    """
    for particle in particles:
        target_box = FMM_search(particle, root)
        target_box.particles.append(particle)
        insert_particle(particle, target_box)

def insert_particle(particle, leaf_box):
    """
    This function adds the particle to the boxes it belongs recursively from the bottom-up. 
    """
    if leaf_box.parent != None:
        leaf_box.parent.particles.append(particle)
        insert_particle(particle, leaf_box.parent)

def FMM_build_fixed_tree(root, particles, levels):
    """
    This function builds the FMM Quadtree with a fixed number of levels from a list of particles given.

    Returns:
    --------
    time_taken : float
        Time taken to build the tree
    """
    start_time = time.perf_counter()
    FMM_create_tree(root, levels)
    FMM_populate_tree(particles, root)
    root.set_Child_Side_Neighbors()
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    return time_taken

################## UPWARD PASS ##################
def multipole(particles, p, center, time_dic): #P2M
    """
    Compute a multipole expansion up to p terms, corresponding to equation (2.3) in Greengard and Rokhlin paper.
    Complex coordinates are used to calculate the multipole expansion coefficients
    Coeffs = [Q, a_k]

    Instead of using a for loop to compute the coefficents, we want to use numpy broadcasting to compute the coefficients faster.
    Using [:, None] to make a vertical vector for broadcasting purposes.

    Returns:
    --------
    a_k : np.array
        Multipole expansion coefficients
    time_dic: dict
        Dictionary of time taken for each operation
    """
    start_time = time.perf_counter()
    a_k = np.zeros(p + 1, dtype=complex)
    if particles:
        q_i = np.array([particle.mass for particle in particles])[:,None]
        z_i = np.array([complex(*particle.pos) - complex(*center) for particle in particles])[:,None]
        k = np.arange(1,p+1)
        a_k[0] = np.sum(q_i)
        a_k[1:] = np.sum(-q_i * z_i**k/k, axis=0) # sum vertically
    end_time = time.perf_counter()
    time_dic["P2M_time"] += end_time - start_time
    return a_k, time_dic

def shift_multipole(a_k, z0, time_dic): #M2M
    """
    Shift multipole expansion to new centre z0, corresponding to equation (2.9) in Greengard and Rokhlin paper.
    shift = [Q, b_l]

    Returns:
    --------
    b_l : np.array
        Shifted multipole expansion coefficients
    time_dic: dict
        Dictionary of time taken for each operation
    """
    start_time = time.perf_counter()
    b_l = np.zeros(len(a_k), dtype=complex)
    b_l[0] = a_k[0]
    for l in range(1, len(a_k)):
        for k in range(1, l+1):
            b_l[l] += a_k[k] * z0**(l - k) * math.comb(l-1, k-1)
        b_l[l] -= (a_k[0] * z0**l)/l
    end_time = time.perf_counter()
    time_dic["M2M_time"] += end_time - start_time
    return b_l, time_dic


def compute_outer_coeffs(box, time_dic):
    """
    In the upward pass, compute outer multipole expansion recursively.
    The outer_coeffs represents the potential field due to all particles contained in one box
   
    For a given box:
    1) If it is a leaf box, compute the multipole expansion of the potential field due to all particles in the box about its centre # P2M
    2) For a parent box, shift the center of each child box's expansion to the current box center and add them together. # M2M 

    Returns:
    --------
    time_dic: dict
        Dictionary of time taken for each operation
    """
    p = box.p
    if not box.children:
       #  Step 1: Form multipole expansions of potential field due to particles in each box about the box center at the finest level. 
        box.outer_coeffs, time_dic = multipole(box.particles, p, box.coords, time_dic)
    else:
        # Step 2: Form multipole expansions about the centers of all boxes at all coarser levels, each expansion representing the potential field due to all particles contained in the box.
        for child in box.children:
            time_dic = compute_outer_coeffs(child, time_dic)
            z0 = complex(*child.coords) - complex(*box.coords) 
            shifted_potential, time_dic = shift_multipole(child.outer_coeffs, z0, time_dic)
            box.outer_coeffs += shifted_potential
    return time_dic

################## DOWNWARD PASS ##################
def multipole_to_local(outer_coeffs, z0, time_dic): #M2L
    """
    Compute the local Taylor expansion from multipole expansion coefficients, corresponding to equation (2.13) and (2.14) in Greengard and Rokhlin paper.
    local = [b_0, b_l]

    Returns:
    --------
    local : np.array
        Local expansion coefficients
    time_dic: dict
        Dictionary of time taken for each operation
    """
    start_time = time.perf_counter()
    local = np.zeros_like(outer_coeffs)
    local[0] = (sum([(outer_coeffs[k]/z0**k)*(-1)**k for k in range(1, len(outer_coeffs))]) +outer_coeffs[0]*np.log(-z0))
    for l in range(1, len(outer_coeffs)):
        temp = complex(0, 0)
        for k in range(1, len(outer_coeffs)):
            temp += (outer_coeffs[k]/z0**k) * math.comb(l+k-1, k-1) * (-1)**k 
        local[l] = (1/z0**l) * temp - outer_coeffs[0]/((z0**l)*l)
    end_time = time.perf_counter()
    time_dic["M2L_time"] += end_time - start_time
    return local, time_dic

def shift_local(coeffs, z0, time_dic): #L2L
    """
    Shift local expansion to new centre z0, corresponding to equation (2.17) in Greengard and Rokhlin paper

    Returns:
    --------
    shift : np.array
        Shifted local expansion coefficients
    time_dic: dict
        Dictionary of time taken for each operation
    """
    start_time = time.perf_counter()
    shift = np.zeros_like(coeffs)
    for l in range(len(coeffs)):
        for k in range(l, len(coeffs)):
            shift[l] += coeffs[k] * math.comb(k,l) * (-z0)**(k-l)
    end_time = time.perf_counter()
    time_dic["L2L_time"] += end_time - start_time
    return shift, time_dic

def compute_inner_coeffs(box, time_dic):
    """
    In the downward pass, compute the inner_coeffs for all cells recursively, from course to fine level.
    The inner_coeffs describes the field due to all particles in the system that are not contained in the current box or its nearest neighbors.

    For a given box:
    1) Local expansion of its parent is shifted to the centre of the currnet box (information flows from parent to child), forming the initial expansion for the box. #L2L
    2) The multipole expansion (outer_coeffs) of the interaction set is converted to local expansion about the centre of the current box # M2L

    Returns:
    --------
    time_dic: dict
        Dictionary of time taken for each operation
    """
    z0 = complex(*box.parent.coords) - complex(*box.coords)
    box.inner_coeffs, time_dic = shift_local(box.parent.inner_coeffs, z0, time_dic)
    for far_box in box.interaction_set():
        z0 = complex(*far_box.coords) - complex(*box.coords)
        interaction_potential, time_dic = multipole_to_local(far_box.outer_coeffs, z0, time_dic)
        box.inner_coeffs += interaction_potential
    return time_dic

################## CALCULATE POTENTIAL ##################
def FMM_calculate_potential_single(box, time_dic):
    """
    This function computes the potential of a box recursively, from course to fine level.

    Returns:
    --------
    time_dic : dict
        Dictionary of time taken for each operation
    """
    time_dic = compute_inner_coeffs(box, time_dic)
    if not box.children: # leaf node
        # Compute potential due to all far enough particles #L2P
        time_dic = evaluatate_local_potential(box, time_dic)
        # Compute potential directly from particles in nearest neighbours #P2P_1
        for n in box.get_Nearest_Neighbours: 
            time_dic = potential_direct_sum_nearest_neighbour(box.particles, n.particles, time_dic)
        # Compute all-to-all potential from all particles in leaf cell #P2P_2
        time_taken = FMM_potential_direct_sum(box.particles)
        time_dic["P2P_time"] += time_taken
    else:
        for child in box.children:
            FMM_calculate_potential_single(child, time_dic)
    return time_dic


def FMM_calculate_potential_all(root):
    """
    This function computes the potential of all particles in the system using the FMM algorithm, while also timing the 6 steps of FMM.
    1. P2M: Compute multipole expansion of the potential field due to all particles in each box about the box center at the finest level (leaf box).
    2. M2M: Shift multipole expansion to the center of the parent box.
    3. M2L: Convert multipole expansion of the interaction set to local expansion about the center of the current box.
    4. L2L: Shift local expansion of the parent box to the center of the current box.
    5. L2P: Compute potential due to all far enough particles, by evaluating the inner cofficients.
    6. P2P: Compute potential directly from particles in nearest neighbours.

    Input:
    ------
    root : Box_fmm
        The root node of the FMM Quadtree

    Returns:
    --------
    time_dic : dict
        Dictionary of time taken for each operation of the form: 
        time_dic = {"P2M_time": float, "M2M_time": float, "M2L_time": float, "L2L_time": float, "L2P_time": float, "P2P_time": float, "fmm_calc": float}
    """
    time_dic = {"P2M_time": 0, "M2M_time": 0, "M2L_time": 0, "L2L_time": 0, "L2P_time": 0, "P2P_time": 0}
    time_dic = compute_outer_coeffs(root, time_dic) # Upward pass
    for child in root.children:
        time_dic = FMM_calculate_potential_single(child, time_dic) # Downward pass and potential evaluation
    time_dic["fmm_calc"] = sum(time_dic.values())
    return time_dic
