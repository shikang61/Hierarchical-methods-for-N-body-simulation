
def BH_insert_particle(box, particle):
    """
    This function inserts a particle into the Barnes-Hut Quadtree.
    """
    if len(box.particles) < box.max_n:
        box.particles.append(particle)
        box.update_Centre_Of_Mass(particle)
        return
    elif not box.children: # If the box has no children, create them and redistribute the particle into the appropriate child box.
        box.children = box.create_Children_Boxes()
        particles = box.particles + [particle]
        child_boxes = [box.get_Child_Box(p) for p in particles]
        for p, child_box in zip(particles, child_boxes):
            BH_insert_particle(child_box, p)
    else: # if the box has children, insert the particle into the appropriate child box.
        child_box = box.get_Child_Box(particle)
        BH_insert_particle(child_box, particle)

