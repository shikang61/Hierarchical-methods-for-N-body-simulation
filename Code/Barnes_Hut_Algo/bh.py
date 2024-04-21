from Supporting_functions.bh_support import distance, bh_potential

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
    if particle not in box.particles:
        box.particles.append(particle)


def BH_build_tree(root, particles):
    """
    This function builds the Barnes-Hut Quadtree from a list of particles given. Insert the particle one by one to build the tree.
    """
    for particle in particles:
        BH_insert_particle(root, particle)
    return root

def BH_calculate_potential_single(particle, box, theta):
    """
    This function calculates the potential at the target particle due to all particles, using the Barnes-Hut approximation.
    """
    if not box.children: # leaf nodes of the Quadtree
        for p in box.particles:
            if p != particle:
                particle.phi += bh_potential(particle, p)
                # print("particle at", p.pos, "contributes potential", potential(particle, box))
    elif quotient(particle, box) < theta:
        # print(quotient(particle, box))
        particle.phi += bh_potential(particle, box)
        # print("box at", box.coords, f"has {box.mass} particles and" ,"contributes potential", potential(particle, box))
    else:
        # print(quotient(particle, box))
        for child in box.children:
            BH_calculate_potential_single(particle, child, theta)
        
    
def BH_calculate_potential_all(particles, root, theta):
    for p in particles:
        print("\n\nnew particle at", p.pos)
        BH_calculate_potential_single(p, root, theta)


def quotient(particle, box):
    """
    This function computes the quotient for the Barnes-Hut approximation.
    """
    return box.size / distance(particle, box)
    



