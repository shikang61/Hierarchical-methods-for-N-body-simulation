import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Barnes_Hut_Algo import BH_build_tree, BH_calculate_potential_all
from FMM_Algo import FMM_build_tree
import numpy as np

fig = plt.figure()
quadt = fig.add_subplot(111, aspect='equal')

def draw_box(box, colour, alpha=1):
    quadt.add_patch(patches.Rectangle((box.coords[0]-box.size/2, box.coords[1]-box.size/2), box.size, box.size, fill = False, color = colour, alpha=alpha))
    
def display(box, colour, alpha):
    if not box.particles:
        return
    if box.children:
        for child in box.children:
            draw_box(child, colour, alpha)
            display(child, colour, alpha)
    else:
        draw_box(box, colour, alpha)

def plot_tree(root, particles, theta, type, animated=False, get_potential=False):
    """
    This function plots the Quadtree.
    """
    if type == "bh":
        BH_build_tree(root, particles)
        max_pos = root.size/2
        ticks = np.arange(-1*max_pos, max_pos+1, step=1)
        if get_potential:
            BH_calculate_potential_all(particles, root, theta)

    elif type == "fmm":
        FMM_build_tree(root, particles)
        max_pos = root.size/2
        ticks = np.arange(-1*max_pos, max_pos+1, step=1)
        root.set_Child_Side_Neighbors()

        # # Visualising intreraction set
        # draw_box(root.children[2].children[2].children[1].children[1], "blue")
        # for i in root.children[2].children[2].children[1].children[1].interaction_set():
        #     if i is not None:
        #         draw_box(i, "red")
        # for i in root.children[2].children[2].children[1].interaction_set():
        #     if i is not None:
        #         draw_box(i, "green")

    if not animated:
        quadt.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=3, c='r')
        plt.xlim(-1*max_pos, max_pos)
        plt.ylim(-1*max_pos,max_pos)
        plt.xticks(ticks)
        plt.yticks(ticks)
        display(root, "black", alpha=0.1)
        plt.show()
    else:
        shown = []
        for p in particles:
            shown.append(p)
            quadt.clear()
            quadt.scatter([shown_p.pos[0] for shown_p in shown], [shown_p.pos[1] for shown_p in shown], s=3, c='r')
            plt.xlim(-1*max_pos, max_pos)
            plt.ylim(-1*max_pos, max_pos)
            plt.xticks(ticks)
            plt.yticks(ticks)
            display(root, "black", alpha=0.1)
            plt.pause(0.05)
    plt.show()



    

