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
N = 1000
theta = 0.5             # separation parameter i.e. degree of approximation for Barnes-Hut

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = False

# Variable parameter: Number of particles
max_leaf = 5
leaf_range = np.arange(1, max_leaf+1)
x_label = r"$n_{leaf}$"
####################################################################

def bh_varying_leaf():
    # Variables to store the results 
    times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []}    # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying N (number of particles) for Barnes-Hut Algorithm **************\n")

    # Print all the simulation parameters
    print(f"""Simulation parameters: \n\t theta = {theta}, N = {N} \n\t Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution}
         Number of data points = {len(leaf_range)} \n\t Range of {x_label} = 1 to {max_leaf} \n""")

    start_time = time.perf_counter()
    for i, n_leaf in enumerate(leaf_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{len(leaf_range)}) Max number of particles at leaf = {n_leaf} ---------\n")
        run_BH_simulation(center, size, n_leaf, theta, particles, N, n_leaf, times, max_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
        
    print("\n ************** Simulation Done ************** \n")

    if save_data:
        if input("Do you want to save the data? (y/n): ") == "y":
            np.save(f"Data/BH_varying_N_{max(leaf_range)}_{theta}_time", times)
            np.save(f"Data/BH_varying_N_{max(leaf_range)}_{theta}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    # Plot log10(time) against log10(n)
    ax = plot_results(leaf_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"BH: $log_{10}$ t vs $log_{10}$ $n_{leaf}$ for " + r"$\theta$ =" f"{theta}")
    plt.savefig("Figure/bh_varying_leaf_logt_vs_logn.png", dpi = 500)

    # Plot t vs N
    ax = plot_results(leaf_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
    ax.set_title(r"BH: t vs $n_{leaf}$ for " + r"$\theta$ =" f"{theta}")
    plt.savefig("Figure/bh_varying_leaf_t_vs_n.png", dpi = 500)

    if direct_sum:
        # Plot log(max_error) against n
        ax = plot_results(leaf_range, max_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(r"BH: max error vs $log_{10}$ $n_{leaf}$ for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_leaf_maxerr_vs_logn.png", dpi = 500)

        # Visualise the spacial error distribution
        n_leaf = 5
        error = all_error[n_leaf]
        root = roots[n_leaf]
        particles = generate_particles(N, size, seed, distribution)
        ax = plot_tree(root, particles, error = error, visualise_error=True)
        ax.set_title(f"Spatial error distribution for BH Algorithm for {x_label} = {theta:.3f} and n_leaf = {n_leaf}, N = {N}")
        plt.savefig("Figure/bh_varying_leaf_spatial_error_distribution.png", dpi = 500)
