"""
The below code runs the FMM algorithm for different p with N = 1000, m = 6.
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
N = 1000                            # Number of particles
m = 6                               # Maximum number of particles in a m box
boundary = None                     # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed"             # Type of FMM tree: "fixed" or "adaptive
max_levels = set_max_levels(N, m)   # Maximum levels of the tree

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = True      # Option to calculate the direct sum

# Variable parameter: Number of terms in multipole
min_p = 3
max_p = 30
p_range = np.arange(min_p, max_p+1)
x_label = "p"
####################################################################


def fmm_varying_p():
    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "P2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    rel_error = [] # list[float]: A list containing the relative error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying p (number of terms) for Fast Multipole Algorithm **************\n")
    print(f"""Simulation parameters: 
         N = {N} \n\t boundary = {"wall" if boundary==None else boundary}, levels = {max_levels}
         tree type = {FMM_tree_type},
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in m box = {m} 
         Number of data points = {len(p_range)}, range of {x_label} = 1 to {max_p} \n""")

    start_time = time.perf_counter()
    for i, p in enumerate(p_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{len(p_range)}) Number of terms = {p} ---------\n")
        run_FMM_simulation(center, size, p, m, boundary, particles, max_levels, N, FMM_tree_type, p, times, rel_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if save_data:
        np.save(f"Data/FMM_varying_p_{max_p}_time_{FMM_tree_type}_{max_levels}_{N}", times)
        np.save(f"Data/FMM_varying_p_{max_p}_error_{FMM_tree_type}_{max_levels}_{N}", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    
    try:
        del times["fmm_direct_sum"]
        del times["fmm_create_tree"]
    except:
        pass
    
    # Plot log(t) against log(N)
    ax = plot_results(p_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"FMM: lg t vs lg p " + f"for N = {N}, m = {m}")
    plt.savefig("Figure/fmm_varying_p_logt_vs_logp.png", dpi = 500)

    if direct_sum:
        # Plot log(rel_error) against lg(p)
        ax = plot_results(p_range, rel_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = True)
        ax.set_title(rf"FMM: relative error vs p for N = {N}, m = {m}")
        plt.savefig("Figure/fmm_varying_p_err_vs_p.png", dpi = 500)
            
        