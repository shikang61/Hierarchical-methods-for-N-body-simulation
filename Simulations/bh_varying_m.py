"""
The below code runs the BH algorithm for different m with fixed theta = 0.5.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_BH_simulation
import matplotlib.pyplot as plt
import time

################## Simulation Parameter ##################
# Fixed Parameters for the algorithm
N = 1000                # Number of particles
theta = 0.5             # separation parameter i.e. degree of approximation for Barnes-Hut

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = False

# Variable parameter: Number of particles
min_m = 1
max_m = 5
m_range = np.arange(min_m, max_m+1)
x_label = "m"
####################################################################

def bh_varying_m():
    # Variables to store the results 
    times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []}    # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
    rel_error = [] # list[float]: A list containing the relative error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying N (number of particles) for Barnes-Hut Algorithm **************\n")
    print(f"""Simulation parameters:
         theta = {theta}, N = {N}
         Size of the simulation area = {size} 
         Seed = {seed}
         Distribution of particles = {distribution}
         Number of data points = {len(m_range)}
         Range of {x_label} = {min_m} to {max_m} \n""")

    start_time = time.perf_counter()
    for i, n_m in enumerate(m_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{len(m_range)}) Max number of particles at m = {n_m} ---------\n")
        run_BH_simulation(center, size, n_m, theta, particles, N, n_m, times, rel_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
        
    print("\n ************** Simulation Done ************** \n")

    if save_data:
        if input("Do you want to save the data? (y/n): ") == "y":
            np.save(f"Data/BH_varying_N_{max(m_range)}_{theta}_time", times)
            np.save(f"Data/BH_varying_N_{max(m_range)}_{theta}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    # Plot log10(time) against log10(n)
    ax = plot_results(m_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"BH: lg t vs lg m for " + r"$\theta$ =" f"{theta}")
    plt.savefig("Figure/bh_varying_m_logt_vs_logn.png", dpi = 500)

    if direct_sum:
        # Plot log(rel_error) against n
        ax = plot_results(m_range, rel_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(r"BH: relative error vs lg m for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_m_maxerr_vs_logn.png", dpi = 500)

        # Visualise the spacial error distribution
        n_m = 5
        error = all_error[n_m]
        root = roots[n_m]
        particles = generate_particles(N, size, seed, distribution)
        ax = plot_tree(root, particles, error = error, visualise_error=True)
        ax.set_title(f"Spatial error distribution for BH Algorithm for {x_label} = {theta:.3f} and n_m = {n_m}, N = {N}")
        plt.savefig("Figure/bh_varying_m_spatial_error_distribution.png", dpi = 500)
