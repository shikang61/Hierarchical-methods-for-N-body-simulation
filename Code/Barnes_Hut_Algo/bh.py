# Barnes-Hut Parameter:
theta = 0.5

def BH_insert_particle(box, particle):
    """
    This function inserts a particle into the Barnes-Hut Quadtree.
    """
    box.update_Centre_Of_Mass(particle) # update the centre of mass of the box after every insertion of a particle
    if len(box.particles) < box.max_n: # if the box has vacancy, insert the particle into the box.
        box.particles.append(particle)
    elif not box.children: # If no vacancy and the box has no children, create 4 children and redistribute the particle into the appropriate child box.
        box.children = box.create_Children_Boxes()
        particles = box.particles + [particle]
        child_boxes = [box.get_Child_Box(p) for p in particles]
        for p, child_box in zip(particles, child_boxes):
            BH_insert_particle(child_box, p)
    else: # If no vacancy and the box has children, insert the particle into the appropriate child box.
        child_box = box.get_Child_Box(particle)
        BH_insert_particle(child_box, particle)


def BH_build_tree(root, particles):
    """
    This function builds the Barnes-Hut Quadtree from a list of particles given.
    """
    for particle in particles:
        BH_insert_particle(root, particle)
    return root

def BH_calculate_potential(particle, box):
    pass
