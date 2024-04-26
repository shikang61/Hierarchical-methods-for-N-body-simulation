# Fast Multipole Algorithm: Experiment with varying p (number of terms in multipole expansion)
"""
The below code runs the FMM algorithm for different p with fixed N = 500 and levels = 5.
We will examine the (1) time taken and (2) error of FMM as a function of p.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_FMM_simulation
import matplotlib.pyplot as plt

################## Simulation Parameter ##################
# Fixed Parameters
N = 200                 # Accuracy of the simulation (FMM)
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
max_levels = 6          # Maximum number of levels in the FMM Quadtree
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive

center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
max_n = 1               # Maximum number of particles in a leaf box
seed = 21               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"

# Variable parameter: Number of terms in multipole
max_p = 4
p_range = np.arange(1, max_p+1)
p_range=[max_p]
x_label = "p"
####################################################################


def fmm_varying_p():

    # Results variable
    times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying p (number of terms) for Fast Multipole Algorithm **************\n")

    ################### SIMULATION ###################
    # Print all the simulation parameters
    print(f"""Simulation parameters: 
         N = {N} \n\t boundary = {"wall" if boundary==None else boundary}, maximum levels = {max_levels}
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {max_n} 
         Number of data points = {len(p_range)} \n\t Range of {x_label} = 1 to {max_p} \n""")
    input("Press Enter to begin simulation...")
    for i, p in enumerate(p_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{len(p_range)}) Number of terms = {p} ---------\n")
        run_FMM_simulation(center, size, p, max_n, boundary, particles, max_levels, N, FMM_tree_type, p, times, max_error, all_error, roots)
        
    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if input("Do you want to save the data? (y/n): ") == "y":
        np.save(f"Data/FMM_varying_p_{max_p}_time_{FMM_tree_type}_{max_levels}_{N}", times)
        np.save(f"Data/FMM_varying_p_{max_p}_error_{FMM_tree_type}_{max_levels}_{N}", all_error)
        print("--- Data saved successfully ---\n")

    ################### GENERATING PLOTS ###################
    plt.close()
    # Plot log2(time) against log2(N)
    if input(f"Do you want to plot the results for time vs {x_label}? (y/n): ") == "y":
        # log2(t) vs log2(p)
        del times["fmm_direct_sum"]
        del times["fmm_create_tree"]
        ax = plot_results(p_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
        ax.set_title(fr"FMM: $log_{2}$ t vs $log_{2}$ p for N = {N}, levels = {max_levels}, tree type = {FMM_tree_type}")
        plt.savefig("Figure/fmm_varying_p_logt_vs_logp.png")

        # t vs p
        ax = plot_results(p_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(f"FMM: t vs p for N = {N}, levels = {max_levels}, tree type = {FMM_tree_type} ")
        plt.savefig("Figure/fmm_varying_p_t_vs_p.png")

    # Plot max_error against p
    if input(f"Do you want to plot the results for error vs {x_label}? (y/n): ") == "y":
        ax = plot_results(p_range, max_error, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(rf"FMM: max error vs p for N = {N}, levels = {max_levels}, tree type = {FMM_tree_type}")
        plt.savefig("Figure/fmm_varying_p_maxerr_vs_p.png")
    
    # Visualise the Quadtree
    if input("Do you want to visualise the Quadtree? (y/n): ") == "y":
        ax = plot_tree(roots[max(p_range)], particles, visualise_interaction=True)
        ax.set_title(rf"Quadtree for FMM Algorithm for  p = {p}, levels = {max_levels}, tree type = {FMM_tree_type} and N = {max(p_range)}")
        plt.savefig("Figure/fmm_varying_p_quadtree_interactions.png")

    # Plot the error distribution
    if input("Do you want to plot the error distribution? (y/n): ") == "y":
        for x, error in zip(p_range[:4:1], list(all_error.values())[:4:1]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title("Error distribution for FMM Algorithm")
            plt.savefig("Figure/fmm_varying_p_error_distribution.png")
        
    # Visualise the error distribution
    if input("Do you want to visualise the spatial error distribution? (y/n): ") == "y":
        print(p_range[::2])
        while True:
            try:
                p_idx = int(input("Enter the index of the value of p you want to visualise: "))
                p = p_range[p_idx]
                break
            except:
                print("invalid input")
                pass
        error = all_error[p]
        ax = plot_tree(roots[p], particles, error = error, visualise_error=True)
        ax.set_title(f"Spatial error distribution for FMM Algorithm for p = {p}, N = {N}, levels = {max_levels}")
        plt.savefig("Figure/fmm_varying_p_spatial_error_distribution.png")
        


    plt.show()