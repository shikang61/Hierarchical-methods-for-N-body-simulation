# Hierarchical methods for N-body simulation
The N-body problem, which consists of calculating pairwise interactions between N bodies, is a well-studied problem in computational physics. A brute-force evaluation of potential would take O(N2) time which quickly becomes impractical for large N. Once it is agreed that the potential is only needed to be calculated up to a certain accuracy, faster approximate algorithms can be used. The Barnes-Hut (BH) algorithm and Fast Multipole Methods (FMM) are two hierarchical methods that are implemented and investigated in this project. Both of them are treecode algorithms which uses the quadtree as a data structure to store all the particles in the system and all the information relevant for potential evaluation. The BH algorithm works by grouping together sufficiently far particles using the separation parameter while the FMM expresses the potential of a group of particles as a truncated multipole expansion. The BH algorithm has a time complexity of O(N log N ) and the FMM has time complexity of O(29Np2/m + 2Np + 9Nm) ∼ O(N).\\

![FMM](https://github.com/shikang61/Hierarchical-methods-for-N-body-simulation/assets/61730647/09665f3a-ebb0-4b3b-b0c0-6fef5fc6578e | width = 100)



![s:d](https://github.com/shikang61/Hierarchical-methods-for-N-body-simulation/assets/61730647/5bfd966f-78b1-4dc7-b953-5df1f538bafe | width = 100)
