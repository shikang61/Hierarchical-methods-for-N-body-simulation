# Barnes-Hut Algorithm: Experiment with varying N (number of particles)
"""
The below code runs the BH algorithm for different N with fixed theta = 0.5.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_BH_simulation
import matplotlib.pyplot as plt
import time

################## Simulation Parameter ##################
# Fixed Parameters for the algorithm
theta = 0.5             # separation parameter i.e. degree of approximation for Barnes-Hut
bh_leaf = 1               # Maximum number of particles in a leaf box

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = True

# Variable parameter: Number of particles
data_points = 30         # Number of data points
max_N = 10000
N_range = np.logspace(2, np.log10(max_N), data_points).astype(int) # [100, 117, 137, 161, 188, 221, 259, 303, 356, 417, 489, 573, 672, 788, 923, 1082, 1268, 1487, 1743, 2043, 2395, 2807, 3290, 3856, 4520, 5298, 6210, 7278, 8531, 10000]
x_label = "N"
####################################################################

def bh_varying_n():
    # Variables to store the results 
    times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []}    # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying N (number of particles) for Barnes-Hut Algorithm **************\n")

    # Print all the simulation parameters
    print(f"""Simulation parameters: \n\t theta = {theta} \n\t Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {bh_leaf} 
         Number of data points = {data_points} \n\t Range of {x_label} = 100 to {max_N} \n""")

    start_time = time.perf_counter()
    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} ---------\n")
        run_BH_simulation(center, size, bh_leaf, theta, particles, N, N, times, max_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
        
    print("\n ************** Simulation Done ************** \n")

    if save_data:
        if input("Do you want to save the data? (y/n): ") == "y":
            np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_time", times)
            np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    # Plot log10(time) against log10(N)
    ax = plot_results(N_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"BH: $log_{10}$ t vs $log_{10}$ N for " + r"$\theta$ =" f"{theta}")
    plt.savefig("Figure/bh_varying_N_logt_vs_logN.png", dpi = 500)

    # Plot t vs N
    ax = plot_results(N_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(r"BH: t vs N for " + r"$\theta$ =" f"{theta}")
    plt.savefig("Figure/bh_varying_N_t_vs_N.png", dpi = 500)

    if direct_sum:
        # Plot max_error against log10(N)
        ax = plot_results(N_range, max_error, x_label, plot_style = "line", log_x = True, log_y = False, fit_line = False)
        ax.set_title(r"BH: max error vs $log_{10}$ N for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_N_maxerr_vs_logN.png", dpi = 500)

        # Visualise the interaction on the Quadtree
        N = 303
        particles = generate_particles(N, size, seed, distribution)
        ax = plot_tree(roots[N], particles, visualise_interaction=True)
        ax.set_title(rf"Quadtree for BH Algorithm for $\theta$ = 0.5 and N = {N}")
        plt.savefig("Figure/bh_varying_N_quadtree_interactions.png", dpi = 500)

        # Plot the error distribution
        count = 1
        for x, error in zip(N_range[::3], list(all_error.values())[::3]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title(fr"Error distribution for BH Algorithm for $\theta$ =" f"{theta}")
            plt.savefig(f"Figure/bh_varying_N_error_distribution_{count}.png", dpi = 500)
            count += 1

        # Visualise the spacial error distribution
        N = 5298
        error = all_error[N]
        root = roots[N]
        particles = generate_particles(N, size, seed, distribution)
        ax = plot_tree(root, particles, error = error, visualise_error=True)
        ax.set_title(f"Spatial error distribution for BH Algorithm for {x_label} = {theta:.3f} and N = {N}")
        plt.savefig("Figure/bh_varying_N_spatial_error_distribution.png", dpi = 500)
