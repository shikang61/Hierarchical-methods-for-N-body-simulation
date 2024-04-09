import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Barnes_Hut_Algo import BH_build_tree, BH_calculate_potential_all
import numpy as np

fig = plt.figure()
quadt = fig.add_subplot(111, aspect='equal')

def draw_box(box):
    quadt.add_patch(patches.Rectangle((box.coords[0]-box.size/2, box.coords[1]-box.size/2), box.size, box.size, fill = False))
    
def display(box):
    if not box.particles:
        return
    if box.children:
        for child in box.children:
            draw_box(child)
            display(child)
    else:
        draw_box(box)

def plot_tree(root, particles, theta, animated=False, get_potential=False):
    """
    This function plots the Barnes-Hut Quadtree.
    """
    BH_build_tree(root, particles)
    max_pos = root.size/2
    ticks = np.arange(-1*max_pos, max_pos+1, step=1)
    if get_potential:
        BH_calculate_potential_all(particles, root, theta)

    if not animated:
        quadt.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=10, c='r')
        plt.xlim(-1*max_pos, max_pos)
        plt.ylim(-1*max_pos,max_pos)
        plt.xticks(ticks)
        plt.yticks(ticks)
        display(root)
        plt.show()
    else:
        shown = []
        for p in particles:
            shown.append(p)
            quadt.clear()
            quadt.scatter([shown_p.pos[0] for shown_p in shown], [shown_p.pos[1] for shown_p in shown], s=5, c='r')
            plt.xlim(-1*max_pos, max_pos)
            plt.ylim(-1*max_pos, max_pos)
            plt.xticks(ticks)
            plt.yticks(ticks)
            display(root)
            plt.pause(0.05)
    plt.show()



    

