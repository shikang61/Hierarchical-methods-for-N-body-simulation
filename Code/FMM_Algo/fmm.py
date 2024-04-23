"""
Implementation of Fast Muilti-pole Method (FMM) algorithm
"""

from itertools import chain
import numpy as np
from .fmm_support import distance, FMM_potential, potential_direct_sum_nearest_neighbour, potential_direct_sum, FMM_search
import math


################## QUADTREE BUILDING ##################
# Adaptive quadtree
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
    """
    for particle in particles:
        FMM_insert_particle(root, particle)
    root.set_Child_Side_Neighbors()
    return root

# Quadtree with a given number of levels
def FMM_create_tree(box, levels):
    if box.level < levels:
        if not box.children:
            box.children = box.create_Children_Boxes()
            for child in box.children:
                FMM_create_tree(child, levels)

def FMM_populate_tree(particles, root):
    for particle in particles:
        target_box = FMM_search(particle, root)
        target_box.particles.append(particle)
        inseet_particle(particle, target_box)

def inseet_particle(particle, leaf_box):
    if leaf_box.parent != None:
        leaf_box.parent.particles.append(particle)
        inseet_particle(particle, leaf_box.parent)
    
def FMM_build_fixed_tree(root, particles, levels):
    FMM_create_tree(root, levels)
    FMM_populate_tree(particles, root)
    root.set_Child_Side_Neighbors()
    return root

################## UPWARD PASS ##################
def multipole(particles, p, center): #S2M

    """
    Compute a multipole expansion up to p terms, corresponding to equation (2.3) in Greengard and Rokhlin paper.
    Complex coordinates are used to calculate the multipole expansion coefficients
    Coeffs = [Q, a_k]

    Instead of using a for loop to compute the coefficents, we want to use numpy broadcasting to compute the coefficients faster.
    Using [:, None] to make a vertical vector for broadcasting purposes.
    """
    a_k = np.zeros(p + 1, dtype=complex)
    if particles:
        q_i = np.array([particle.mass for particle in particles])[:,None]
        z_i = np.array([complex(*particle.pos) - complex(*center) for particle in particles])[:,None]
        k = np.arange(1,p+1)
        a_k[0] = np.sum(q_i)
        a_k[1:] = np.sum(-q_i * z_i**k/k, axis=0) # sum vertically
    return a_k

def shift_multipole(a_k, z0): #M2M
    """
    Shift multipole expansion to new centre z0, corresponding to equation (2.9) in Greengard and Rokhlin paper.
    shift = [Q, b_l]
    """
    b_l = np.zeros(len(a_k), dtype=complex)
    b_l[0] = a_k[0]
    for l in range(1, len(a_k)):
        for k in range(1, l+1):
            b_l[l] += a_k[k] * z0**(l - k) * math.comb(l-1, k-1)
        b_l[l] -= (a_k[0] * z0**l)/l
    return b_l

def compute_outer_coeffs(box):
    """
    In the upward pass, compute outer multipole expansion recursively.
    The outer_coeffs represents the potemial field due to all particles contained in one box
   
    For a given box:
    1) If it is a leaf box, compute the multipole expansion of the potential field due to all particles in the box about its centre # S2M
    2) For a given box, shift the center of each child box's expansion to the current box center and add them together. # M2M 
    """
    p = box.p
    if not box.children:
       #  Step 1: Form multipole expansions of potential field due to particles in each box about the box center at the finest level. 
        box.outer_coeffs = multipole(particles=box.particles, p=p, center=box.coords)
    else:
        # Step 2: Form multipole expansions about the centers of all boxes at all coarser levels, each expansion representing the potential field due to all particles contained in the box.
        for child in box.children:
            compute_outer_coeffs(child)
            z0 = complex(*child.coords) - complex(*box.coords) 
            box.outer_coeffs += shift_multipole(child.outer_coeffs, z0)

################## DOWNWARD PASS ##################
def multipole_to_local(outer_coeffs, z0): #M2L
    """
    Compute the local Taylor expansion from multipole expansion coefficients, corresponding to equation (2.13) and (2.14) in Greengard and Rokhlin paper.
    local = [b_0, b_l]
    """
    local = np.zeros_like(outer_coeffs)
    local[0] = (sum([(outer_coeffs[k]/z0**k)*(-1)**k for k in range(1, len(outer_coeffs))]) +outer_coeffs[0]*np.log(-z0))
    for l in range(1, len(outer_coeffs)):
        temp = complex(0, 0)
        for k in range(1, len(outer_coeffs)):
            temp += (outer_coeffs[k]/z0**k) * math.comb(l+k-1, k-1) * (-1)**k 
        local[l] = (1/z0**l) * temp - outer_coeffs[0]/((z0**l)*l)
    return local

def shift_local(coeffs, z0): #L2L
    """
    Shift local expansion to new centre z0, corresponding to equation (2.17) in Greengard and Rokhlin paper
    """
    shift = np.zeros_like(coeffs)
    for l in range(len(coeffs)):
        for k in range(l, len(coeffs)):
            shift[l] += coeffs[k] * math.comb(k,l) * (-z0)**(k-l)
    return shift

def compute_inner_coeffs(box):
    """
    In the downward pass, compute the inner_coeffs for all cells recursively, from course to fine level.
    The inner_coeffs describes the field due to all particles in the system that are not contained in the current box or its nearest neighbors.

    For a given box:
    1) Local expansion of its parent is shifted to the centre of the currnet box (information flows from parent to child), forming the initial expansion for the box. #L2L
    2) The multipole expansion (outer_coeffs) of the interaction set is converted to local expansion about the centre of the current box # M2L
    """
    z0 = complex(*box.parent.coords) - complex(*box.coords)
    box.inner_coeffs = shift_local(box.parent.inner_coeffs, z0)
    for far_box in box.interaction_set():
        z0 = complex(*far_box.coords) - complex(*box.coords)
        box.inner_coeffs += multipole_to_local(far_box.outer_coeffs, z0)

################## CALCULATE POTENTIAL ##################
def FMM_calculate_potential_single(box):
    compute_inner_coeffs(box)
    if not box.children: # leaf node
        # Compute potential due to all far enough particles
        z0 = complex(*box.coords)
        coeffs = box.inner_coeffs
        
        for particle in box.particles:
            z = complex(*particle.pos)
            particle.phi += np.real(np.polyval(coeffs[::-1], z-z0))

        # Compute potential directly from particles in nearest neighbours
        for n in box.get_Nearest_Neighbours:
            potential_direct_sum_nearest_neighbour(box.particles, n.particles)

        # Compute all-to-all potential from all particles in leaf cell
        potential_direct_sum(box.particles)
    else:
        for child in box.children:
            FMM_calculate_potential_single(child)

def FMM_calculate_potential_all(root):
    compute_outer_coeffs(root)
    for child in root.children:
        FMM_calculate_potential_single(child)
