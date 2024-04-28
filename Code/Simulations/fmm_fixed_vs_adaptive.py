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
fmm_leaf = 5            # Maximum number of particles in a leaf box

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area

seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
direct_sum = False

# Variable parameter: Number of particles
data_points = 20         # Number of data points
max_N = 50000
N_range = np.logspace(2, np.log10(max_N), data_points).astype(int)
x_label = "N"
FMM_tree_type_1 = "fixed" 
FMM_tree_type_2 = "adaptive"

####################################################################

def fmm_fixed_vs_adaptive():
    # Results variable
    times_fixed = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error_fixed = [] # list[float]: A list containing the max error for each N
    all_error_fixed = {} # list[list[float]]: A list containing the error of every particle for each N
    roots_fixed = {}

    times_adaptive = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error_adaptive = [] # list[float]: A list containing the max error for each N
    all_error_adaptive = {} # list[list[float]]: A list containing the error of every particle for each N
    roots_adaptive = {}
    max_relative_error = []
    
    print("\n ************** Start experiment with varying N (number of particles) for Fast Multipole Algorithm **************\n")

    # Print all the simulation parameters
    print(f"""Simulation parameters: 
         p = {p}, boundary = {"wall" if boundary==None else boundary}, maximum levels = $log_4$(N)
         Maximum number of particles in leaf box = {fmm_leaf} 
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} 
         Number of data points = {len(N_range)} \n\t Range of {x_label} = 100 to {max(N_range)} \n""")

    start_time = time.perf_counter()
    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)
        if N == 266:
            particle_keep = particles
        max_levels = set_max_levels(N, fmm_leaf) # dynamically set max_levels based on N
        print(f"\n--------- ({i+1}/{len(N_range)}) Number of particles = {N} ---------\n")
        phi_fixed = run_FMM_simulation(center, size, p, fmm_leaf, boundary, particles, max_levels, N, FMM_tree_type_1, N, times_fixed, max_error_fixed, all_error_fixed, roots_fixed, direct_sum)
        phi_adaptive = run_FMM_simulation(center, size, p, fmm_leaf, boundary, particles, max_levels, N, FMM_tree_type_2, N, times_adaptive, max_error_adaptive, all_error_adaptive, roots_adaptive, direct_sum)
        relative_error = calculate_error(phi_adaptive, phi_fixed)
        max_relative_error.append(np.max(relative_error))

    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")

    plt.close()
    times_fixed["fmm_create_tree_fixed"] = times_fixed.pop("fmm_create_tree")
    times_adaptive["fmm_create_tree_adaptive"] = times_adaptive.pop("fmm_create_tree")
    times_fixed["fmm_calc_fixed"] = times_fixed.pop("fmm_calc")
    times_adaptive["fmm_calc_adaptive"] = times_adaptive.pop("fmm_calc")
    all_times = {**times_fixed, **times_adaptive}
    keys = ["fmm_create_tree_fixed", "fmm_create_tree_adaptive", "fmm_calc_fixed", "fmm_calc_adaptive"]
    time_to_plot = {key: all_times[key] for key in keys}

    # logt vs logN
    ax = plot_results(N_range, time_to_plot, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"FMM: $log_{10}$ t vs $log_{10}$ N" + f" for p = {p} " +  r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_N_logt_vs_logN_fixed_adaptive.png", dpi = 500)
    
    # # t vs N
    ax = plot_results(N_range, time_to_plot, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(f"FMM: t vs N for p = {p} " +  r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_N_t_vs_N_fixed_adaptiv.png", dpi = 500)

    # Plot log(relative error) against logN
    ax = plot_results(N_range, max_relative_error, x_label, plot_style = "line", log_x = True, log_y = True, fit_line = False, error_label="_FMM fixed vs FMM_adaptive")
    ax.set_title(r"Max relative error between FMM (fixed vs adaptive) vs $log_{10}$ N for " + f"p = {p} "+ r"$leaf_{fmm}$ " + f"= {fmm_leaf}")
    plt.savefig("Figure/bh_vs_fmm_varying_N_error_vs_logN.png", dpi = 500)    
    
    # Visualise the Quadtree
    N = 266
    ax = plot_tree(roots_adaptive[N], particle_keep, visualise_interaction=True)
    ax.set_title(rf"Quadtree for FMM Algorithm for  p = {p}, tree type = {FMM_tree_type_1}, N = {N}, " + r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_N_quadtree_interactions_adaprive.png", dpi = 500)