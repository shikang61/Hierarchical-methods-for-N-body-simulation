import numpy as np

class Box():
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
            Box((x0-size/2, y0+size/2), size, self, max_n=self.max_n),
            Box((x0+size/2, y0+size/2), size, self, max_n=self.max_n),
            Box((x0-size/2, y0-size/2), size, self, max_n=self.max_n),
            Box((x0+size/2, y0-size/2), size, self, max_n=self.max_n)
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

    

        
            