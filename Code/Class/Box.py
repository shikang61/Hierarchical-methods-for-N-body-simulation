import numpy as np
from Supporting_functions.fmm_support import get_neighbours_child, get_corner_neighbours

class Box_bh():
    """
    The Box class acts as the Nodes for a (Quad)tree. It is used to store the particles in the Quadtree.
    It contains information about the centre_of_mass and total_mass of all particles in contains.
    It is connected to a parent node and can have 4 children nodes.
    Q0  |  Q1
    -----------
    Q2  |  Q3
    The maximum number of particles allowed in a Box is defined by the max_n parameter.

    Initialisation parameters:
    ---------
    pos: array
        The position of the Box
    size: int, powers of 2 ideally 
        The size of the Box
    max_n: int
        The maximum number of particles allowed in the Box
    parent: 
        The parent of the Box


    Other attributes:
    ---------
    pos: array
        The centre of mass of the Box
    mass: float
        The total mass of the Box
    """

    def __init__(self, coords, size, parent, max_n=1):
        self.coords = np.array(coords)
        self.size = size
        self.max_n = max_n
        self.parent = parent

        self.children = []
        self.particles = []
        self.pos = self.coords
        self.mass = 0

    def create_Children_Boxes(self):
        """
        This method creates and returns the 4 children boxes of the current box.
        Q0  |  Q1
        ----------
        Q2  |  Q3
        """
        x0, y0 = self.coords
        size = self.size / 2
        return [
            Box_bh((x0-size/2, y0+size/2), size, self, max_n=self.max_n),
            Box_bh((x0+size/2, y0+size/2), size, self, max_n=self.max_n),
            Box_bh((x0-size/2, y0-size/2), size, self, max_n=self.max_n),
            Box_bh((x0+size/2, y0-size/2), size, self, max_n=self.max_n)
        ]
    def get_Child_Box(self, particle):
        """
        This method returns the quadrant in which the particle belongs.
        Q0  |  Q1
        ----------
        Q2  |  Q3
        """
        x0, y0 = self.coords
        x, y = particle.pos
        return self.children[[[2, 3], [0, 1]][int(y > y0)][int(x > x0)]]
    
    def update_Centre_Of_Mass(self, particle):
        """
        This method updates the centre of mass of the Box after adding a particle.
        """
        x_com, y_com = self.pos
        M = self.mass
        x, y = particle.pos
        m = particle.mass
        x_new, y_new = (x_com*M + x*m)/(M+m), (y_com*M + y*m)/(M+m)

        self.pos = np.array([x_new, y_new])
        self.mass += m

    
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
    def __init__(self, coords, size, c_index, parent=None, level=0, max_n=1):
        self.coords = np.array(coords)
        self.size = size
        self.parent = parent
        self.max_n = max_n
          
        self.particles = []
        self.children = []
        self.side_neighbours = 4*[None,]
        self.nearest_neighbours = None
        self.c_index = c_index

        self.bottom_left = self.coords - self.size/2
        self.level = level
        self.inner, self.outer = None, None

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
            Box_fmm((x0-size/2, y0+size/2), size, 0, self, level = self.level+1, max_n=self.max_n),
            Box_fmm((x0+size/2, y0+size/2), size, 1, self, level = self.level+1, max_n=self.max_n),
            Box_fmm((x0-size/2, y0-size/2), size, 2, self, level = self.level+1, max_n=self.max_n),
            Box_fmm((x0+size/2, y0-size/2), size, 3, self, level = self.level+1, max_n=self.max_n)
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
    


        



        

    

    

    
