import numpy as np

class Box:
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
    centre_of_mass: array
        The centre of mass of the Box
    total_mass: float
        The total mass of the Box
    """

    def __init__(self, coords, size, parent, max_n=1):
        self.coords = np.array(coords)
        self.size = size
        self.max_n = max_n
        self.parent = parent

        self.children = []
        self.particles = []
        self.centre_of_mass = self.pos
        self.total_mass = 0

    def create_Children_Boxes(self):
        """
        This method creates and returns the 4 children boxes of the current box.
        Q0  |  Q1
        ----------
        Q2  |  Q3
        """
        if not self.children:
            x0, y0 = self.coords
            size = self.size / 2
            return [
                Box((x0-size/2, y0+size/2), size, self, max_n=self.max_n),
                Box((x0+size/2, y0+size/2), size, self, max_n=self.max_n),
                Box((x0-size/2, y0-size/2), size, self, max_n=self.max_n),
                Box((x0+size/2, y0-size/2), size, self, max_n=self.max_n)
        ]
    
    def update_Centre_Of_Mass(self, particle):
        """
        This method updates the centre of mass of the Box after adding a particle.
        """
        x_com, y_com = self.centre_of_mass
        M = self.total_mass
        x, y = particle.pos
        m = particle.mass
        x_new, y_new = (x_com*M + x*m)/(M+m), (y_com*M + y*m)/(M+m)

        self.centre_of_mass = np.array([x_new, y_new])
        self.total_mass += m

    def within_Box(self, particle):
        """
        This method checks if a particle is within the Box.
        """
        x0, y0 = self.coords
        x, y = particle.pos
        size = self.size
        return (x0-size/2 <= x <= x0+size/2) and (y0-size/2 <= y <= y0+size/2)
    
    def add_Particle(self, particle):
        """
        This method adds a particle to the Box.
        If the number of particles in the Box exceeds the max_n parameter, it creates children boxes and distributes the particles among them.
        """
        self.particles.append(particle)
        if len(self.particles) <= self.max_n:    
            self.update_Centre_Of_Mass(particle)
        elif self.children: # if children boxes already exist, add the new particle to the appropriate child box
            for child in self.children:
                if child.within_Box(particle):
                    child.add_Particle(particle)
                    break
        else: # if children boxes do not exist, create them and distribute the particles among them
            self.children = self.create_Children_Boxes()
            all_particles = self.particles + [particle]
            for particle in all_particles:
                for child in self.children:
                    if child.within_Box(particle):
                        child.add_Particle(particle)
                        break
        
            