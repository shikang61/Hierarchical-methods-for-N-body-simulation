def BH_insert_particle(box, particle):
    """
    This function inserts a particle into the Barnes-Hut Quadtree.
    """
    if len(box.particles) < box.max_n:
        box.particles.append(particle)
        box.update_Centre_Of_Mass(particle)
        return

    elif 
    