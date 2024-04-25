import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .general import search
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

def plot_tree(quadtree, particles, animated=False, visualise_interaction=False):
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

    Returns:
    --------
    ax : matplotlib axis
        return the axis of the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    
    max_pos = quadtree.size/2

    # Visualising intreraction set
    if visualise_interaction and quadtree.type == "fmm":
        target = input("Enter the particle index to visualise its interaction set: ")
        if target == "":
            target = np.random.choice(np.range(1, len(particles)+1))

        ax.scatter([p.pos[0] for p in particles if p.idx in target], [p.pos[1] for p in particles if p.idx in target], s=10, c='b')
        find = [p for p in particles if p.idx in target]
        for p in find:
            target_box = search(quadtree, p)
            for i in target_box.get_Nearest_Neighbours:
                if i is not None:
                    draw_box(ax, i, "red")

    # Plotting the quadtree, either animatedor not
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
        ax.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=1, c='r')
        plt.xlim(-1*max_pos, max_pos)
        plt.ylim(-1*max_pos,max_pos)
        display(ax, quadtree, "black", alpha=0.01)

    return ax
