# Fast Multipole Algorithm: Experiment with varying N (number of particles)
"""
The below code runs the FMM algorithm for different N with fixed p = 7 and levels = 5.
We will examine the (1) time taken and (2) error of FMM as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_FMM_simulation
import matplotlib.pyplot as plt

################## Simulation Parameter ##################
# Fixed Parameters
p = 7                   # Number of terms in multipole (FMM)
N = 300
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive

center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
max_n = 1               # Maximum number of particles in a leaf box
seed = 21               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"

# Variable parameter: Number of particless
max_levels = 6
lvl_range = np.arange(1, max_levels+1)
x_label = "max levels"
####################################################################

def fmm_varying_levels():

    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying N (number of particles) for Fast Multipole Algorithm **************\n")

    ################### SIMULATION ###################
    # Print all the simulation parameters
    print(f"""Simulation parameters: 
         p = {p}, boundary = {"wall" if boundary==None else boundary}, Number of particles = {N}
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {max_n} 
         Number of data points = {len(lvl_range)} \n\t Range of {x_label} = 1 to {max_levels} \n""")
    input("Press Enter to begin simulation...")

    for i, lvl in enumerate(lvl_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{len(lvl_range)}) Number of levels = {lvl} ---------\n")
        run_FMM_simulation(center, size, p, max_n, boundary, particles, lvl, N, FMM_tree_type, lvl, times, max_error, all_error, roots)
        
    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if input("Do you want to save the data? (y/n): ") == "y":
        np.save(f"Data/FMM_varying_levels_{max(lvl_range)}_time_{p}_{FMM_tree_type}_{N}", times)
        np.save(f"Data/FMM_varying_levels_{max(lvl_range)}_error_{p}_{FMM_tree_type}_{N}", all_error)
        print("--- Data saved successfully ---\n")

    ################### GENERATING PLOTS ###################
    plt.close()
    # Plot log2(time) against levels
    if input(f"Do you want to plot the results for time vs {x_label}? (y/n): ") == "y":
        # log2(t) vs levels
        ax = plot_results(lvl_range, times, x_label, plot_style = "scatter", log_x = False, log_y = True, fit_line = False)
        ax.set_title(fr"FMM: $log_{2}$ t vs max levels for p = {p}, N = {N}, tree type = {FMM_tree_type}")
        plt.savefig("Figure/fmm_varying_level_logt_vs_levelsN.png")

        # t vs levels
        ax = plot_results(lvl_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(f"FMM: t vs levels for p = {p}, N = {N}, tree type = {FMM_tree_type} ")
        plt.savefig("Figure/fmm_varying_level_t_vs_level.png")

    # Plot max_error against levels
    if input(f"Do you want to plot the results for error vs {x_label}? (y/n): ") == "y":
        ax = plot_results(lvl_range, max_error, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(rf"FMM: max error vs max levels for p = {p}, N = {N}, tree type = {FMM_tree_type}")
        plt.savefig("Figure/fmm_varying_level_maxerr_vs_levels.png")
    
    # Visualise the Quadtree
    if input("Do you want to visualise the Quadtree? (y/n): ") == "y":
        ax = plot_tree(roots[max(lvl_range)], particles, visualise_interaction=True)
        ax.set_title(rf"Quadtree for FMM Algorithm for  p = {p}, N = {N}, tree type = {FMM_tree_type} and levels = {max(lvl_range)}")
        plt.savefig("Figure/fmm_varying_level_quadtree_interactions.png")

    # Plot the error distribution
    if input("Do you want to plot the error distribution? (y/n): ") == "y":
        for x, error in zip(lvl_range[:4:1], list(all_error.values())[:4:1]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title("Error distribution for FMM Algorithm")
            plt.savefig("Figure/fmm_varying_level_error_distribution.png")

    # Visualise the error distribution
    if input("Do you want to visualise the spatial error distribution? (y/n): ") == "y":
        print(lvl_range[::2])
        while True:
            try:
                lvl_idx = int(input("Enter the index of the value of theta you want to visualise: "))
                lvl = lvl_range[lvl_idx]
                break
            except:
                print("invalid input")
                pass
        error = all_error[lvl]
        particles = generate_particles(N, size, seed, distribution)
        ax = plot_tree(roots[lvl], particles, error = error, visualise_error=True)
        ax.set_title(f"Spatial error distribution for FMM Algorithm for N = {N}, p = {p}, levels = {lvl}")
        plt.tight_layout()
        plt.savefig("Figure/fmm_varying_level_spatial_error_distribution.png")

    plt.show()