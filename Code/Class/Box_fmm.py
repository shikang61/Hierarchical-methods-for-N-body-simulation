import numpy as np
from FMM_Algo.fmm_support import get_neighbours_child, get_corner_neighbours
import math

class Box_fmm():
    """
    The Box class acts as the Nodes for a Quadtree

    Initialisation parameters:
    ---------

    Other attributes:
    ---------
    c_index: int
        The index of the box in the parent's children list
    """
    def __init__(self, coords, size, p=5, c_index=0, parent = None, level=0, max_n=1, boundary=None):
        self.coords = np.array(coords)
        self.size = size
        self.bottom_left = self.coords - self.size/2
        self.parent = parent
        self.max_n = max_n
        self.p = p
          
        self.particles = []
        self.children = []
        self.side_neighbours = 4*[None]
        if boundary == "periodic" and level == 0: # Only for root node
            self.side_neighbours = 4*[self]
        self.nearest_neighbours = None
        self.c_index = c_index
        
        self.level = level
        self.inner_coeffs = np.zeros((self.p + 1), dtype=complex)
        self.outer_coeffs = np.zeros((self.p + 1), dtype=complex)

    def create_Children_Boxes(self):
        """
        This method creates and returns the 4 children boxes of the current box.
         0 | 1
        -------
         2 | 3
        """
        x0, y0 = self.coords
        size = self.size / 2
        return [
            Box_fmm((x0-size/2, y0+size/2), size, p = self.p, c_index = 0, parent = self, level = self.level+1, max_n=self.max_n),
            Box_fmm((x0+size/2, y0+size/2), size, p = self.p,  c_index = 1, parent = self, level = self.level+1, max_n=self.max_n),
            Box_fmm((x0-size/2, y0-size/2), size, p = self.p, c_index = 2, parent = self, level = self.level+1, max_n=self.max_n),
            Box_fmm((x0+size/2, y0-size/2), size, p = self.p, c_index = 3, parent = self, level = self.level+1, max_n=self.max_n)
        ]
    
    def get_Child_Box(self, particle):
        """
        This method returns the quadrant in which the particle belongs.
         0 | 1
        -------
         2 | 3
        """
        x0, y0 = self.coords
        x, y = particle.pos
        return self.children[[[2, 3], [0, 1]][int(y > y0)][int(x > x0)]]
    
    def get_Child_At_Index(self, i):
        if len(self.children) == 0:
            return self
        else:
            return self.children[i]
    
    def set_Child_Side_Neighbors(self):
        """
        This method sets the neighbors of the children of the current box. A side neighbour is a box that shares a side with the current box.   
        """
        for i, child in enumerate(self.children):
            """
             0 | 1
            -------
             2 | 3
            Sibling neighbours: 0 -> (1,2), 1 -> (0,3), 2 -> (0,3), 3 -> (1,2) i.e (i^1, i^2)

            Index ordering of side neighbours: 0 (West), 1 (North), 2 (East), 3 (South)
            E.g. Position index of side neighbours:
            |   | 1 |   |   |
            ----------------
            | 0 | * |       |
            --------|   2   |
            |   | 3 |       |
            """
            # Set sibling neighbors
            sibling_neighbours_index = (abs(1 + (i^1) - i), abs(1 + (i^2) - i))
            child.side_neighbours[sibling_neighbours_index[0]] = self.children[i^1]
            child.side_neighbours[sibling_neighbours_index[1]] = self.children[i^2]

            """
                | 2 | 3 |   |
            ----*********----
              1 [ 0 | 1 ] 0 |
            -----------------
              3 [ 2 | 3 ] 2 |
            -----********-----
                | 0 | 1 |   |
            Child of parent's neighbours.
            """
            # Set the other side neighbours from parent's neighbours
            remaining_side_neighbours_index = tuple(set((0,1,2,3)) - set((sibling_neighbours_index)))
            j, k = remaining_side_neighbours_index
            child.side_neighbours[j] = get_neighbours_child(self, i, j, k)
            child.side_neighbours[k] = get_neighbours_child(self, i, k, j)

            # Recursively set side neighbours
            if child.children is not None:
                child.set_Child_Side_Neighbors()
    
    @property
    def get_Nearest_Neighbours(self):
        if self.nearest_neighbours is not None:
            return self.nearest_neighbours
        # Get all corner neighbours
        corner_neighbours = get_corner_neighbours(self)
        nearest_neighbours = [n for n in list(set(self.side_neighbours + corner_neighbours)) if n is not None]
        self.nearest_neighbours = nearest_neighbours
        return nearest_neighbours

    def interaction_set(self):
        nn = self.get_Nearest_Neighbours
        pn = self.parent.get_Nearest_Neighbours
        int_set = []
        for n in pn:
            if len(n.children) != 0:
                int_set += [c for c in n.children if c not in nn]
            elif n not in nn:
                int_set.append(n)
        return int_set