# Barnes-Hut Algorithm: Experiment with varying N (number of particles)
"""
The below code runs the BH algorithm for different N with fixed theta = 0.5.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_BH_simulation
import matplotlib.pyplot as plt

################## Simulation Parameter ##################
# Fixed Parameters
theta = 0.5             # separation parameter i.e. degree of approximation for Barnes-Hut
center = (0, 0)         # Center of the simulation area
size = 1024              # Size of the simulation area
max_n = 1               # Maximum number of particles in a leaf box
seed = 21               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"

# Variable parameter: Number of particles
data_points = 5         # Number of data points
max_N = 500
N_range = np.logspace(2, np.log10(max_N), data_points).astype(int)
x_label = "N"
####################################################################



def bh_varying_n():

    # Results variable
    times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []}    # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
    max_error = [] # list[float]: A list containing the max error for each N
    all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    roots = {}

    print("\n ************** Start experiment with varying N (number of particles) for Barnes-Hut Algorithm **************\n")

    ################### SIMULATION ###################
    # Print all the simulation parameters
    print(f"""Simulation parameters: \n\t theta = {theta} \n\t Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {max_n} 
         Number of data points = {data_points} \n\t Range of {x_label} = 100 to {max_N} \n""")
    input("Press Enter to begin simulation...")

    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} ---------\n")
        run_BH_simulation(center, size, max_n, theta, particles, N, N, times, max_error, all_error, roots)
        
    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if input("Do you want to save the data? (y/n): ") == "y":
        np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_time", times)
        np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_error", all_error)
    print("--- Data saved successfully ---\n")

    ################### GENERATING PLOTS ###################
    plt.close()
    # Plot log2(time) against log2(N)
    if input(f"Do you want to plot the results for time vs N? (y/n): ") == "y":
        # log2(t) vs log2(N)
        ax = plot_results(N_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
        ax.set_title(r"BH: $log_{2}$ t vs $log_{2}$ N for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_N_logt_vs_logN.png")

        # t vs N
        ax = plot_results(N_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(r"BH: t vs N for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_N_t_vs_N.png")

    # Plot max_error against log2(N)
    if input(f"Do you want to plot the results for error vs N? (y/n): ") == "y":
        ax = plot_results(N_range, max_error, x_label, plot_style = "line", log_x = True, log_y = False, fit_line = False)
        ax.set_title(r"BH: max error vs $log_{2}$ N for " + r"$\theta$ =" f"{theta}")
        plt.savefig("Figure/bh_varying_N_maxerr_vs_logN.png")
    
    # Visualise the Quadtree
    if input("Do you want to visualise the Quadtree? (y/n): ") == "y":
        ax = plot_tree(roots[max(N_range)], particles, visualise_interaction=True)
        ax.set_title(rf"Quadtree for BH Algorithm for $\theta$ = 0.5 and N = {max(N_range)}")
        plt.savefig("Figure/bh_varying_N_quadtree_interactions.png")

    # Plot the error distribution
    if input("Do you want to plot the error distribution? (y/n): ") == "y":
        for x, error in zip(N_range[:4:1], list(all_error.values())[:4:1]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title("Error distribution for BH Algorithm")
            plt.savefig("Figure/bh_varying_N_error_distribution.png")

    plt.show()