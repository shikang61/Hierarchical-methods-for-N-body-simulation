import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Class import Box
from Barnes_Hut_Algo import BH_build_tree

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

def plot_tree(particles, animated=False):
    """
    This function plots the Barnes-Hut Quadtree.
    """
    root = Box((0, 0), 2, None)
    if not animated:
        quadt.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=10, c='r')
        BH_build_tree(root, particles)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        display(root)
        plt.show()
    else:
        shown = []
        for p in particles:
            shown.append(p)
            quadt.clear()
            quadt.scatter([shown_p.pos[0] for shown_p in shown], [shown_p.pos[1] for shown_p in shown], s=5, c='r')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            BH_build_tree(root, [p])
            display(root)
            plt.pause(0.05)
    plt.show()
    