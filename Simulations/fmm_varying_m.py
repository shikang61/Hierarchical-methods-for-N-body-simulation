"""
The below code runs the FMM algorithm for different m with fixed p = 7 and dynamic levels
We will examine the (1) time taken and (2) error of FMM as a function of m.
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
direct_sum = False      # Option to calculate the direct sum

# Variable parameter: Number of particles in leaf box
min_m = 5
max_m = 19
m_range = np.arange(min_m, max_m+1) # so that l = 5
x_label = "m"
####################################################################

def fmm_varying_m():
    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "P2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    rel_error = [] # list[float]: A list containing the relative error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}
    
    print("\n ************** Start experiment with varying number of m particles for Fast Multipole Algorithm **************\n")
    print(f"""Simulation parameters: 
         N = {N}, 
         p = {p}, boundary = {"wall" if boundary==None else boundary}, levels = log_4(N/m)
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} 
         Number of data points = {len(m_range)}, range of {x_label} = 1 to {max(m_range)} \n""")

    start_time = time.perf_counter()
    for i, m in enumerate(m_range):
        particles = generate_particles(N, size, seed, distribution)
        max_levels = set_max_levels(N, m) # dynamically set max_levels based on N
        print(f"\n--------- ({i+1}/{len(m_range)}) Number of particles at m = {m} ---------\n")
        run_FMM_simulation(center, size, p, m, boundary, particles, max_levels, N, FMM_tree_type, m, times, rel_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")

    plt.close()

    keys = ["fmm_calc", "P2P_time", "M2L_time"]
    time_to_expt = {key: np.array(times[key]) for key in keys}
    ax = plot_results(m_range, time_to_expt, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(r"FMM: t vs m "+ f"for p = {p}, tree type = {FMM_tree_type}, N = {N}")
    plt.savefig("Figure/fmm_varying_m_t_vs_m_particular.png", dpi = 500)

    # Plot log10(time) against log10(m)
    ax = plot_results(m_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"FMM: lg t vs lg m " + f"for p = {p}, N = {N}")
    plt.savefig("Figure/fmm_varying_m_logt_vs_logm.png", dpi = 500)

    # # t vs m
    ax = plot_results(m_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(r"FMM: t vs m" + f" for p = {p}, N = {N}" )
    plt.savefig("Figure/fmm_varying_m_t_vs_m.png", dpi = 500)

    if direct_sum:
        # Plot log(rel_error) against m
        ax = plot_results(m_range, rel_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(r"FMM: relative error vs m" + f"for N= {N}")
        plt.savefig("Figure/fmm_varying_m_maxerr_vs_m.png", dpi = 500)

    # Conclusion: levels is the more important factor in determining the accuracy