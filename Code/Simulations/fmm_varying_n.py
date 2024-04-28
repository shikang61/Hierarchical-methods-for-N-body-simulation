# Fast Multipole Algorithm: Experiment with varying N (number of particles)
"""
The below code runs the FMM algorithm for different N with fixed p = 6 and dynamic levels
We will examine the (1) time taken and (2) error of FMM as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_FMM_simulation
import matplotlib.pyplot as plt
import time
from FMM_Algo import set_max_levels

################## Simulation Parameter ##################
# Fixed Parameters for the algorithm
p = 6                   # Number of terms in multipole (FMM)
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive
fmm_leaf = 5            # Maximum number of particles in a leaf box

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area

seed = 30               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = False

# Variable parameter: Number of particles
data_points = 30         # Number of data points
max_N = 100000
N_range = np.logspace(2, np.log10(max_N), data_points).astype(int) #[100, 143, 206, 297, 428, 615, 885, 1274, 1832, 2636, 3792, 5455, 7847, 11288, 16237, 23357, 33598, 48329, 69519, 100000]
x_label = "N"

####################################################################

def fmm_varying_n():
    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}
    
    print("\n ************** Start experiment with varying N (number of particles) for Fast Multipole Algorithm **************\n")

    # Print all the simulation parameters
    print(f"""Simulation parameters: 
         p = {p}, boundary = {"wall" if boundary==None else boundary}, maximum levels = $log_4$(N)
         tree type = {FMM_tree_type},
         Maximum number of particles in leaf box = {fmm_leaf} 
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} 
         Number of data points = {len(N_range)} \n\t Range of {x_label} = 100 to {max(N_range)} \n""")

    start_time = time.perf_counter()
    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)
        max_levels = set_max_levels(N, fmm_leaf) # dynamically set max_levels based on N
        print(f"\n--------- ({i+1}/{len(N_range)}) Number of particles = {N} ---------\n")
        run_FMM_simulation(center, size, p, fmm_leaf, boundary, particles, max_levels, N, FMM_tree_type, N, times, max_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")

    # # Option to save simulation data
    if save_data:
        np.save(f"Data/FMM_varying_N_{max(N_range)}_time_{p}_{FMM_tree_type}_{max_levels}", times)
        np.save(f"Data/FMM_varying_N_{max(N_range)}_error_{p}_{FMM_tree_type}_{max_levels}", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()

    # Plot log10(time) against log10(N)
    ax = plot_results(N_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"FMM: $log_{10}$ t vs $log_{10}$ N " + f"for p = {p}, tree type = {FMM_tree_type}, " +  r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_N_logt_vs_logN.png", dpi = 500)

    # t vs N
    ax = plot_results(N_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(f"FMM: t vs N for p = {p}, tree type = {FMM_tree_type}, " +  r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_N_t_vs_N.png", dpi = 500)
    
    # Visualise the Quadtree
    N = N_range[5]
    ax = plot_tree(roots[N], particles, visualise_interaction=True)
    ax.set_title(rf"Quadtree for FMM Algorithm for  p = {p}, tree type = {FMM_tree_type}, N = {N}, " + r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_N_quadtree_interactions.png", dpi = 500)