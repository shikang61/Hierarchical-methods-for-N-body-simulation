from .Box import Box

class Canvas():
    """
    The Canvas class is an object which the particles are placed. It is the root node of the QuadTree.
    It contains methods that allows for the visualisation of the particles and Boxes.



    Initialisation paramets:
    ---------

    Other attributes:
    ---------
    """
    def __init__(self, coords, size, particles, max_n = 1):
        self.coords = coords
        self.size = size
        self.max_n = max_n
        self.rootBox = Box(coords, size, None, max_n)
        self.particles = particles
     
    def draw(self):
        pass

    def draw_Boxes(self):
        pass    