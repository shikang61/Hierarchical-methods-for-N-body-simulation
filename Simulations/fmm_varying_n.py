"""
The below code runs the FMM algorithm for different N with fixed p = 6, m = 5 and dynamic levels
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
m = 5                   # Maximum number of particles in a m box
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = True       # Option to calculate the direct sum

# Variable parameter: Number of particles
data_points = 20         # Number of data points
min_N = 100
if direct_sum:
    max_N = 12000
else:
    max_N = 100000
N_range = np.logspace(np.log10(min_N), np.log10(max_N), data_points).astype(int)
x_label = "N"
####################################################################

def fmm_varying_n():
    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "P2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    rel_error = [] # list[float]: A list containing the relative error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}
    
    print("\n ************** Start experiment with varying N for Fast Multipole Algorithm **************\n")
    print(f"""Simulation parameters: 
         p = {p}, boundary = {"wall" if boundary==None else boundary}, levels ~ log4(N/m)
         Maximum number of particles in m box = {m} 
         Size of the simulation area = {size} 
         Seed = {seed}
         Distribution of particles = {distribution} 
         Number of data points = {len(N_range)}, range of {x_label} = {min_N} to {max_N} \n""")

    start_time = time.perf_counter()
    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)
        max_levels = set_max_levels(N, m) # dynamically set max_levels based on N
        if i == 4:
            particle_i = particles
        print(f"\n--------- ({i+1}/{len(N_range)}) N = {N} ---------\n")
        run_FMM_simulation(center, size, p, m, boundary, particles, max_levels, N, FMM_tree_type, N, times, rel_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if save_data:
        np.save(f"Data/FMM_varying_N_{max(N_range)}_time_{p}_{FMM_tree_type}_{max_levels}", times)
        np.save(f"Data/FMM_varying_N_{max(N_range)}_error_{p}_{FMM_tree_type}_{max_levels}", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()

    # Plot logt against logN
    ax = plot_results(N_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(f"FMM: lg t vs lg N for p = {p}, m = {m}")
    plt.savefig("Figure/fmm_varying_N_logt_vs_logN.png", dpi = 500)

    # lgt against N
    ax = plot_results(N_range, times, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
    ax.set_title(f"FMM: logt vs N for p = {p}, m = {m}")
    plt.savefig("Figure/fmm_varying_N_logt_vs_N.png", dpi = 500)

    if direct_sum:
        # logt vs N
        keys = ["fmm_direct_sum", "fmm_calc"]
        calc_time = {key: np.array(times[key]) for key in keys}
        ax = plot_results(N_range, calc_time, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(f"FMM: logt vs N for p = {p}, m = {m}")
        plt.savefig("Figure/fmm_varying_N_logt_vs_N_2.png", dpi = 500)

        # Plot log(rel_error) against N
        ax = plot_results(N_range, rel_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(f"FMM: relative error vs N for p = {p}, m = {m}")
        plt.savefig("Figure/fmm_varying_N_err_vs_N.png", dpi = 500)
   
    # Visualise the Quadtree
    try: 
        N = N_range[4]
    except: 
        N = N_range[0]
    ax = plot_tree(roots[N], particle_i, visualise_interaction=True)
    ax.set_title(f"Quadtree for FMM Algorithm for  p = {p}, N = {N}, m = {m}")
    plt.savefig("Figure/fmm_varying_N_quadtree_interactions.png", dpi = 500)