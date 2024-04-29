# Hierarchical-methods-for-body-simulation
The N-body problem, which consists of calculating pairwise interactions between N bodies, is a well-studied problem in computational physics. A brute-force evaluation of potential would take O(N2) time which quickly becomes impractical for large N. Once it is agreed that the potential is only needed to be calculated up to a certain accuracy, faster approximate algorithms can be used. The Barnes-Hut (BH) algorithm and Fast Multipole Methods (FMM) are two hierarchical methods that are implemented and investigated in this paper. Both of them are treecode algorithms which uses the quadtree as a data structure to store all the particles in the system and all the information relevant for potential evaluation. The BH algorithm works by grouping together sufficiently far particles using the separation parameter while the FMM expresses the potential of a group of particles as a truncated multipole expansion. The BH algorithm has a time complexity of O(N log N ) and the FMM has time complexity of O(29Np2/m + 2Np + 9Nm) ∼ O(N), both of which are verified through computational experiments. The behaviour and errors of BH algorithm for different N and separation parameter was tested. The runtime and error of FMM for different N, numbers of terms of multipole expansion and number of levels of the quadtree is tested, and shows good agreement with theory.
