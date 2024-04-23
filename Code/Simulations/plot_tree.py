import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Barnes_Hut_Algo import BH_build_tree, BH_calculate_potential_all
from FMM_Algo import FMM_build_adaptive_tree, FMM_build_fixed_tree, FMM_calculate_potential_all
from FMM_Algo.fmm_support import potential_direct_sum
from Supporting_functions.general import reset_particle, search
import numpy as np

fig = plt.figure()
quadt = fig.add_subplot(111, aspect='equal')

def draw_box(box, colour, alpha=1):
    quadt.add_patch(patches.Rectangle((box.coords[0]-box.size/2, box.coords[1]-box.size/2), box.size, box.size, fill = False, color = colour, alpha=alpha))
    
def display(box, colour, alpha):
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

        phi1 = np.array([round(p.phi, 7) for p in particles])
        reset_particle(particles)
        potential_direct_sum(particles)
        phi2 = np.array([round(p.phi, 7) for p in particles])
        diff = np.abs((phi2-phi1)/phi2)
        

    elif type == "fmm":
        FMM_build_adaptive_tree(root, particles)
        # FMM_build_fixed_tree(root, particles, 5)
        max_pos = root.size/2
        # ticks = np.linspace(-1*max_pos, max_pos+1, num=20, endpoint=True, dtype=int)

        if get_potential:
            FMM_calculate_potential_all(root)
            phi1 = np.array([round(p.phi, 7) for p in particles])
            # print(phi1)
            reset_particle(particles)
            potential_direct_sum(particles)
            phi2 = np.array([round(p.phi, 7) for p in particles])
            # print(phi2)
            diff = np.abs((phi2-phi1)/phi2)
        # print(len(root.children[2].particles))
        target = [4]
        
        # Visualising intreraction set
        find = [p for p in particles if p.idx in target]
        for p in find:
            target_box = search(root, p)
            for i in target_box.interaction_set():
                if i is not None:
                    draw_box(i, "red")
            # for i in target_box.parent.interaction_set():
            #     if i is not None:
            #         draw_box(i, "purple")
            # for i in target_box.parent.parent.interaction_set():
            #     if i is not None:
            #         draw_box(i, "blue")
        


    if not animated:
        quadt.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=2, c='r')
        # quadt.scatter([p.pos[0] for p in particles if p.idx in target], [p.pos[1] for p in particles if p.idx in target], s=10, c='b')
        plt.xlim(-1*max_pos, max_pos)
        plt.ylim(-1*max_pos,max_pos)
        # plt.xticks(ticks)
        # plt.yticks(ticks)
        display(root, "black", alpha=0.05)
        plt.show()
    else:
        shown = []
        for p in particles:
            shown.append(p)
            quadt.clear()
            quadt.scatter([shown_p.pos[0] for shown_p in shown], [shown_p.pos[1] for shown_p in shown], s=2, c='r')
            plt.xlim(-1*max_pos, max_pos)
            plt.ylim(-1*max_pos, max_pos)
            # plt.xticks(ticks)
            # plt.yticks(ticks)
            display(root, "black", alpha=0.05)
            plt.pause(0.5)

    plt.scatter(np.arange(1,len(diff)+1), diff)
    plt.show()


    

