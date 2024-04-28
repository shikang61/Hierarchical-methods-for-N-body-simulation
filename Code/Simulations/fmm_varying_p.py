# Fast Multipole Algorithm: Experiment with varying p (number of terms in multipole expansion)
"""
The below code runs the FMM algorithm for different p with fixed N = 500 and levels = 5.
We will examine the (1) time taken and (2) error of FMM as a function of p.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_FMM_simulation
import matplotlib.pyplot as plt
import time
from FMM_Algo import set_max_levels
################## Simulation Parameter ##################
# Fixed Parameters for the algorithm
N = 1000               # Number of particles
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive
fmm_leaf = 5               # Maximum number of particles in a leaf box
max_levels = set_max_levels(N, fmm_leaf) # Maximum levels of the tree

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = False

# Variable parameter: Number of terms in multipole
max_p = 35
p_range = np.arange(1, max_p+1)
x_label = "p"
####################################################################


def fmm_varying_p():
    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying p (number of terms) for Fast Multipole Algorithm **************\n")

    # Print all the simulation parameters
    print(f"""Simulation parameters: 
         N = {N} \n\t boundary = {"wall" if boundary==None else boundary}, levels = {max_levels}
         tree type = {FMM_tree_type},
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {fmm_leaf} 
         Number of data points = {len(p_range)} \n\t Range of {x_label} = 1 to {max_p} \n""")

    start_time = time.perf_counter()
    for i, p in enumerate(p_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{len(p_range)}) Number of terms = {p} ---------\n")
        run_FMM_simulation(center, size, p, fmm_leaf, boundary, particles, max_levels, N, FMM_tree_type, p, times, max_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if save_data:
        np.save(f"Data/FMM_varying_p_{max_p}_time_{FMM_tree_type}_{max_levels}_{N}", times)
        np.save(f"Data/FMM_varying_p_{max_p}_error_{FMM_tree_type}_{max_levels}_{N}", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    # Plot log(time) against log(N)
    del times["fmm_direct_sum"]
    del times["fmm_create_tree"]
    ax = plot_results(p_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"FMM: $log_{10}$ t vs $log_{10}$ p " + f"for N = {N}, tree type = {FMM_tree_type}, " + r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_p_logt_vs_logp.png", dpi = 500)

    # t vs p
    ax = plot_results(p_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(f"FMM: t vs p for N = {N}, tree type = {FMM_tree_type}, " + r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_p_t_vs_p.png", dpi = 500)

    # log(t) vs p
    ax = plot_results(p_range, times, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
    ax.set_title(r"FMM: $log_{10}$ t " + f"vs p for N = {N}, tree type = {FMM_tree_type}, " + r"$n_{leaf}$ = " + f"{fmm_leaf}")
    plt.savefig("Figure/fmm_varying_p_logt_vs_p.png", dpi = 500)

    if direct_sum:
        # Plot log(max_error) against p
        ax = plot_results(p_range, max_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(rf"FMM: max error vs p for N = {N}, tree type = {FMM_tree_type}, " + r"$n_{leaf}$ = " + f"{fmm_leaf}")
        plt.savefig("Figure/fmm_varying_p_maxerr_vs_p.png", dpi = 500)
        
        # Plot the error distribution
        count = 1
        for x, error in zip(p_range[::4], list(all_error.values())[::4]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title(f"Error distribution for FMM Algorithm for N = {N}, tree type = {FMM_tree_type}, " + r"$n_{leaf}$ = " + f"{fmm_leaf}")
            plt.savefig(f"Figure/fmm_varying_p_error_distribution_{count}.png", dpi = 500)
            count += 1
            
        # Visualise the error distribution
        p_list = [10,15,20,30]
        count = 1
        for p in p_list:
            error = all_error[p]
            ax = plot_tree(roots[p], particles, error = error, visualise_error=True)
            ax.set_title(f"Spatial error distribution for FMM Algorithm for p = {p}, N = {N}, levels = {max_levels}")
            plt.savefig(f"Figure/fmm_varying_p_spatial_error_distribution_{count}.png")
            count += 1
        