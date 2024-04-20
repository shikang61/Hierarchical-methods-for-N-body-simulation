"""
Implementation of Fast Muilti-pole Method (FMM) algorithm
"""

from itertools import chain
import numpy as np

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


def FMM_build_tree(root, particles):
    """
    This function builds the Barnes-Hut Quadtree from a list of particles given.
    """
    for particle in particles:
        FMM_insert_particle(root, particle)
    return root

