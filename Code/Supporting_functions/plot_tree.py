import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .general import search, interaction_line
import numpy as np

def draw_box(ax, box, colour, alpha=1):
    ax.add_patch(patches.Rectangle((box.coords[0]-box.size/2, box.coords[1]-box.size/2), box.size, box.size, fill = False, color = colour, alpha=alpha))
    
def display(ax, box, colour, alpha):
    if box.children:
        for child in box.children:
            draw_box(ax, child, colour, alpha)
            display(ax, child, colour, alpha)
    else:
        draw_box(ax, box, colour, alpha)

def plot_tree(quadtree, particles, error=None, animated=False, visualise_interaction=False, visualise_error=False):
    """
    This function visualises the Quadtree.

    Inputs:
    ---------
    quadtree: Box
        The root of the Quadtree
    particles: list
        The list of particles
    animated: bool
        If True, the plot will be animated
    visualise_interaction: bool
        If True, the interaction set of a particle will be visualised (for FMM only)
    visualise_com: bool
        If True, the center of mass of each box will be visualised (for BH only)

    Returns:
    --------
    ax : matplotlib axis
        return the axis of the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    max_pos = quadtree.size/2

    # Visualising intreraction set (FMM only)
    if visualise_interaction and quadtree.type == "fmm":
        target = np.random.choice(np.arange(1, len(particles)+1))
        target_particle = [p for p in particles if p.idx == target][0]
        ax.scatter([p.pos[0] for p in particles if p == target_particle], [p.pos[1] for p in particles if p == target_particle], s=10, c='black')
        target_box = search(quadtree, target_particle )
        for i in target_box.interaction_set():
            if i is not None:
                draw_box(ax, i, "red")
        if target_box.parent is not None:
            for i in target_box.parent.interaction_set():
                if i is not None:
                    draw_box(ax, i, "blue")
        if target_box.parent.parent is not None:
            for i in target_box.parent.parent.interaction_set():
                    if i is not None:
                        draw_box(ax, i, "orange")
        
    # Visualising the interaction of the particle with the centre of masses (BH only)
    if visualise_interaction and quadtree.type == "bh":
        max_mass = quadtree.mass
        target = np.random.choice(np.arange(1, len(particles)+1))
        target_particle = [p for p in particles if p.idx == target][0]
        ax.scatter([p.pos[0] for p in particles if p == target_particle], [p.pos[1] for p in particles if p == target_particle], s=25, c='black')
        interaction_line(ax, target_particle, quadtree, max_mass)

    # Plotting the quadtree, either animated or not
    if animated:
        shown = []
        for p in particles:
            shown.append(p)
            ax.clear()
            ax.scatter([shown_p.pos[0] for shown_p in shown], [shown_p.pos[1] for shown_p in shown], s=1, c='r')
            plt.xlim(-1*max_pos, max_pos)
            plt.ylim(-1*max_pos, max_pos)
            display(ax, quadtree, "black", alpha=0.01)
            plt.pause(0.01)
    else:
        # Visualising the error distribution on the Quadtree
        if visualise_error and error is not None:
            im = ax.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=3, c=error, cmap='cividis_r', vmin=min(error), vmax=max(error))
            fig.colorbar(im, label= '(abs.) fractional error', fraction=0.03, pad=0.1)

        else:
            ax.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=1, c='r')
        plt.xlim(-1*max_pos, max_pos)
        plt.ylim(-1*max_pos,max_pos)
        display(ax, quadtree, "black", alpha=0.01)
    return ax
