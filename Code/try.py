import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Class import *
from Barnes_Hut_Algo import *


fig2 = plt.figure()
quadt = fig2.add_subplot(111, aspect='equal')

def plot(box):
    quadt.add_patch(patches.Rectangle((box.coords[0]-box.size/2, box.coords[1]-box.size/2), box.size, box.size, fill = False))


def display(root):
    if not root.particles:
        return
    if root.children:
        for child in root.children:
            plot(child)
            display(child)
    else:
        plot(root)

def main():
    root = Box((0, 0), 2, None)
    # Generate a list of randomly placed particles of length 20
    particles = [Particle((np.random.uniform(-1, 1), np.random.uniform(-1, 1)), 1) for _ in range(100)]
    quadt.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], s=10, c='r')
    BH_build_tree(root, particles)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    display(root)
    plt.show()

main()
