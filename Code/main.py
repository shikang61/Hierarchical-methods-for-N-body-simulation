from Simulations import *

################## "Default Simulation Parameter ##################
# theta = 0.8             # for Barnes-Hut
# center = (0, 0)         # Center of the simulation area
# N = 7000                 # Number of particles
# size = 128              # Size of the simulation area
# max_n = 1               # Maximum number of particles in a leaf box
# seed = 21               # Seed for random number generator
# distribution = "random" # Distribution of the particles: "random" or "triangular"
# particles = generate_particles(N, size, seed, distribution)

# FMM
p = 7                  # Accuracy of the simulation for FMM
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition
levels = 3              # Maxsimum number of levels in the FMM Quadtree
####################################################################


print("start")
# bh_varying_n()
bh_varying_theta()
# # root = Box_bh(coords = center, 
# #               size = size, 
# #               max_n = max_n,
# #               theta = theta)
# root = Box_fmm(coords = center, 
#                size = size, 
#                p = p, 
#                max_n = max_n,
#                boundary = boundary)
# tree_time = FMM_build_fixed_tree(root, particles, levels)
# # tree_time = BH_build_tree(root, particles)
# print(f"Time taken to build the tree: {tree_time:.4f} seconds")
# # potential_time = BH_calculate_potential_all(particles, root)
# potential_time = FMM_calculate_potential_all(root)
# phi_bh = get_potential(particles)
# print(f"Time taken to calculate the potential: {sum(list(potential_time.values())):.4f} seconds")
# # direct_time = BH_potential_direct_sum(particles)
# direct_time = FMM_potential_direct_sum(particles)
# phi_direct = get_potential(particles)
# print(f"Time taken to calculate the potential directly: {direct_time:.4f} seconds")
# print(np.mean(np.abs((phi_direct - phi_bh)/phi_direct)))
# plot_tree(root, particles, animated=False, visualise_interaction=False)



