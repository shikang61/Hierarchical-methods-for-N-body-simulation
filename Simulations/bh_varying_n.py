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
theta = 0.5         # separation parameter i.e. degree of approximation for Barnes-Hut
m = 1               # Maximum number of particles in a m box

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = False

# Variable parameter: Number of particles
data_points = 30         
min_N = 100
max_N = 10000
N_range = np.logspace(np.log10(min_N), np.log10(max_N), data_points).astype(int) # [100, 117, 137, 161, 188, 221, 259, 303, 356, 417, 489, 573, 672, 788, 923, 1082, 1268, 1487, 1743, 2043, 2395, 2807, 3290, 3856, 4520, 5298, 6210, 7278, 8531, 10000]
x_label = "N"
####################################################################

def bh_varying_n():
    # Variables to store the results 
    times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []}  # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
    rel_error = [] # list[float]: A list containing the relative error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying N (number of particles) for Barnes-Hut Algorithm **************\n")
    print(f"""Simulation parameters:
         theta = {theta}
         Size of the simulation area = {size} 
         Seed = {seed} 
         Distribution of particles = {distribution} 
         Maximum number of particles in m box = {m} 
         Number of data points = {data_points}, range of {x_label} = {min_N} to {max_N} \n""")

    start_time = time.perf_counter()
    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} ---------\n")
        run_BH_simulation(center, size, m, theta, particles, N, N, times, rel_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
        
    print("\n ************** Simulation Done ************** \n")

    if save_data:
        if input("Do you want to save the data? (y/n): ") == "y":
            np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_time", times)
            np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    # Plot log(time) against log(N)
    ax = plot_results(N_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"BH: lg t vs lg N for " + r"$\theta$ =" f"{theta}")
    plt.savefig("Figure/bh_varying_N_logt_vs_logN.png", dpi = 500)

    # Plot log10(time) against N
    ax = plot_results(N_range, times, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
    ax.set_title(r"BH: lg t vs N for " + r"$\theta$ =" f"{theta}")
    plt.savefig("Figure/bh_varying_N_logt_vs_N.png", dpi = 500)

    # Visualise the interaction on the Quadtree
    N = N_range[0]
    particles = generate_particles(N, size, seed, distribution)
    ax = plot_tree(roots[N], particles, visualise_interaction=True)
    ax.set_title(rf"Quadtree for BH Algorithm for $\theta$ = 0.5 and N = {N}")
    plt.savefig("Figure/bh_varying_N_quadtree_interactions.png", dpi = 500)


    if direct_sum:
        # Plot rel_error against log10(N)
        ax = plot_results(N_range, rel_error, x_label, plot_style = "line", log_x = True, log_y = False, fit_line = False)
        ax.set_title(r"BH: relative error vs lg N for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_N_maxerr_vs_logN.png", dpi = 500)

        # Plot lg(rel_error) against N
        ax = plot_results(N_range, rel_error, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
        ax.set_title(r"BH: lg(relative error) vs N for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_N_lgmaxerr_vs_N.png", dpi = 500)

        # Plot lg(rel_error) against lgN
        ax = plot_results(N_range, rel_error, x_label, plot_style = "line", log_x = True, log_y = True, fit_line = True)
        ax.set_title(r"BH: lg(relative error) vs N for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_N_lgmaxerr_vs_lgN.png", dpi = 500)

        # Plot the error distribution
        count = 1
        for x, error in zip(N_range[::3], list(all_error.values())[::3]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title(fr"Error distribution for BH Algorithm for $\theta$ =" f"{theta}")
            plt.savefig(f"Figure/bh_varying_N_error_distribution_{count}.png", dpi = 500)
            count += 1

        # Visualise the spacial error distribution
        try:
            N = 259
            error = all_error[N]
            root = roots[N]
            particles = generate_particles(N, size, seed, distribution)
            ax = plot_tree(root, particles, error = error, visualise_error=True)
            ax.set_title(f"Spatial error distribution for BH Algorithm for {x_label} = {theta:.3f} and N = {N}")
            plt.savefig("Figure/bh_varying_N_spatial_error_distribution.png", dpi = 500)
        except:
            pass
