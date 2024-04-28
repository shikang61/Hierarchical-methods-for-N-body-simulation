# Fast Multipole Algorithm: Experiment with varying N (number of particles)
"""
The below code runs the FMM algorithm for different N with fixed p = 7 and levels = 5.
We will examine the (1) time taken and (2) error of FMM as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_FMM_simulation
import matplotlib.pyplot as plt
import time

################## Simulation Parameter ##################
# Fixed Parameters for the algorithm
p = 6                   # Number of terms in multipole (FMM)
N = 5000                # Number of particles
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive
fmm_leaf = 5            # Maximum number of particles in a leaf box
direct_sum = False

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data

# Variable parameter: Number of particless
max_levels = 7
lvl_range = np.arange(2, max_levels+1)
x_label = "lvl"
####################################################################

def fmm_varying_levels():

    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying number of levels for Fast Multipole Algorithm **************\n")

    # Print all the simulation parameters
    print(f"""Simulation parameters: 
         p = {p}, boundary = {"wall" if boundary==None else boundary}, Number of particles = {N}
         tree type = {FMM_tree_type},
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {fmm_leaf} 
         Number of data points = {len(lvl_range)} \n\t Range of {x_label} = 2 to {max_levels} \n""")
    
    start_time = time.perf_counter()
    for i, lvl in enumerate(lvl_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{len(lvl_range)}) Number of levels = {lvl} ---------\n")
        run_FMM_simulation(center, size, p, fmm_leaf, boundary, particles, lvl, N, FMM_tree_type, lvl, times, max_error, all_error, roots, direct_sum)
        if lvl == 5:
            particles_5 = particles
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
        
    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if save_data:
        np.save(f"Data/FMM_varying_levels_{max(lvl_range)}_time_{p}_{FMM_tree_type}_{N}", times)
        np.save(f"Data/FMM_varying_levels_{max(lvl_range)}_error_{p}_{FMM_tree_type}_{N}", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    try:
        del times["fmm_direct_sum"]
    except:
        pass
    # Plot log10(time) against levels
    # ax = plot_results(lvl_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    # ax.set_title(r"FMM: $log_{10}$ t "+ f"vs levels for p = {p}, N = {N}, tree type = {FMM_tree_type}")
    # plt.savefig("Figure/fmm_varying_level_logt_vs_levels.png", dpi = 500)

    # t vs levels
    keys = ["fmm_calc", "P2P_time", "M2L_time"]
    time_to_expt = {key: np.array(times[key]) for key in keys}
    ax = plot_results(lvl_range, time_to_expt, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(f"FMM: t vs levels for p = {p}, N = {N}, tree type = {FMM_tree_type} ")
    plt.savefig("Figure/fmm_varying_level_t_vs_level.png", dpi = 500)

    if direct_sum:
        # Plot max_error against levels
        ax = plot_results(lvl_range, max_error, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(rf"FMM: max error vs levels for p = {p}, N = {N}, tree type = {FMM_tree_type}")
        plt.savefig("Figure/fmm_varying_level_maxerr_vs_levels.png", dpi = 500)
        
        # Visualise the Quadtree
        lvl = 5
        ax = plot_tree(roots[lvl], particles_5, visualise_interaction=True)
        ax.set_title(f"Quadtree for FMM Algorithm for  p = {p}, N = {N}, tree type = {FMM_tree_type} and levels = {lvl}")
        plt.savefig("Figure/fmm_varying_level_quadtree_interactions.png", dpi = 500)

        # Plot the error distribution
        count = 1
        for x, error in zip(lvl_range[::2], list(all_error.values())[::2]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title(f"Error distribution for FMM Algorithm for p = {p}, N = {N}, tree type = {FMM_tree_type}")
            plt.savefig(f"Figure/fmm_varying_level_error_distribution_{count}.png", dpi = 500)
            count += 1

        lvl = 5
        error = all_error[lvl]
        particles = generate_particles(N, size, seed, distribution)
        ax = plot_tree(roots[lvl], particles, error = error, visualise_error=True)
        ax.set_title(f"Spatial error distribution for FMM Algorithm for N = {N}, p = {p}, levels = {lvl}")
        plt.savefig("Figure/fmm_varying_level_spatial_error_distribution.png", dpi = 500)