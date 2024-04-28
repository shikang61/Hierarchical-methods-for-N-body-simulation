# Fast Multipole Algorithm: Experiment with varying number of particles at leaf
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
N = 5000                # Number of particles
p = 7                   # Number of terms in multipole (FMM)
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area

seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = False

# Variable parameter: Number of particles

max_leaf = 20
leaf_range = np.arange(1, max_leaf+1)
x_label = r"$n_{leaf}$"

####################################################################

def fmm_varying_leaf():
    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}
    
    print("\n ************** Start experiment with varying number of leaf particles for Fast Multipole Algorithm **************\n")

    # Print all the simulation parameters
    print(f"""Simulation parameters: 
         N = {N}, tree type = {FMM_tree_type}
         p = {p}, boundary = {"wall" if boundary==None else boundary}, maximum levels = $log_4$(N/n_leaf)
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} 
         Number of data points = {len(leaf_range)} \n\t Range of {x_label} = 1 to {max(leaf_range)} \n""")

    start_time = time.perf_counter()
    for i, fmm_leaf in enumerate(leaf_range):
        particles = generate_particles(N, size, seed, distribution)
        max_levels = set_max_levels(N, fmm_leaf) # dynamically set max_levels based on N
        print(f"\n--------- ({i+1}/{len(leaf_range)}) Number of particles at leaf = {fmm_leaf} ---------\n")
        run_FMM_simulation(center, size, p, fmm_leaf, boundary, particles, max_levels, N, FMM_tree_type, fmm_leaf, times, max_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")

    plt.close()
    try:
        del times["fmm_direct_sum"]
    except:
        pass
    keys = ["fmm_calc", "P2P_time", "M2L_time"]
    time_to_expt = {key: np.array(times[key]) for key in keys}
    ax = plot_results(leaf_range, time_to_expt, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(r"FMM: t vs $n_{leaf}$ " + f"for p = {p}, tree type = {FMM_tree_type}, N = {N}")
    plt.savefig("Figure/fmm_varying_leaf_t_vs_leaf_particular.png", dpi = 500)

    # # Plot log10(time) against log10(leaf)
    # ax = plot_results(leaf_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    # ax.set_title(r"FMM: $log_{10}$ t vs $log_{10}$ $n_{leaf}$ " + f"for p = {p}, tree type = {FMM_tree_type}, N = {N}")
    # plt.savefig("Figure/fmm_varying_leaf_logt_vs_log(leaf).png", dpi = 500)

    # # # t vs leaf
    # ax = plot_results(leaf_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    # ax.set_title(r"FMM: t vs $n_{leaf}$ " + f" for p = {p}, tree type = {FMM_tree_type}, N = {N}" )
    # plt.savefig("Figure/fmm_varying_leaf_t_vs_leaf.png", dpi = 500)

    if direct_sum:
        # Plot log(max_error) against leaf
        ax = plot_results(leaf_range, max_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(r"FMM: max error vs $n_{leaf}$ " + f"for N= {N}, tree type = {FMM_tree_type}")
        plt.savefig("Figure/fmm_varying_leaf_maxerr_vs_leaf.png", dpi = 500)


    # Conclusion: levels is the more important factor in determining the accuracy