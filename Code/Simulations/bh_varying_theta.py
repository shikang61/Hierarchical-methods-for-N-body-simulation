# Barnes-Hut Algorithm: Experiment with varying θ (separation parameter for Barnes-Hut Algorithm)
"""
The below code runs the BH algorithm for different θ with fixed N = 500.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_BH_simulation
import matplotlib.pyplot as plt
import time

################## Simulation Parameter ##################
# Fixed Parameters for the algorithm
N = 5000                # Number of particles
bh_leaf = 1               # Maximum number of particles in a leaf box

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = True

# Variable parameter: separation parameter θ 
data_points = 35         # Number of data points
max_theta = 19
theta_range = np.linspace(0.01, max_theta, data_points) 
x_label = r"$\theta$"
####################################################################

def bh_varying_theta():

    # Results variable
    times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []} # dict[str, list[float]]: A dictionary containing the time taken for each step for each θ
    max_error = []  # list[float]: A list containing the max error for each θ
    all_error = {}  # list[list[float]]: A list containing the error of every particle for each θ
    roots = {}

    print("\n ************** Start experiment with varying θ (separation parameter) for Barnes-Hut Algorithm **************\n")
    
    # Print all the simulation parameters
    print(f"""Simulation parameters: \n\t Number of particles = {N} \n\t Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {bh_leaf} 
         Number of data points = {data_points} \n\t Range of {x_label} = 0.01 to {max_theta} \n""")
    
    start_time = time.perf_counter()
    for i, theta in enumerate(theta_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{data_points}) theta = {theta} ---------\n")
        run_BH_simulation(center, size, bh_leaf, theta, particles, N, theta, times, max_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")
    
    # # Option to save simulation data
    if save_data:
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_time", times)
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    # Plot log(time) against theta
    ax = plot_results(theta_range, times, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)  
    ax.set_title(r"BH: $log_{10}$ t "+ f"vs {x_label} for " + r"N = " f"{N}")
    plt.savefig("Figure/bh_varying_theta_logt_vs_theta.png", dpi = 500)
        
    # t vs θ
    ax = plot_results(theta_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)  
    ax.set_title(f"BH: t vs {x_label} for N = " f"{N}")
    plt.savefig("Figure/bh_varying_theta_t_vs_theta.png", dpi = 500)

    if direct_sum:
        # Plot max_error against theta
        ax = plot_results(theta_range, max_error, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(f"BH: max error vs {x_label} for N =" f"{N}")    
        plt.savefig("Figure/bh_varying_theta_maxerr_vs_theta.png", dpi = 500)

        # Plot error distribution
        count = 1
        for x, error in zip(theta_range[::3], list(all_error.values())[::3]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title(f"Error distribution for BH Algorithm for N = {N}")
            plt.savefig(f"Figure/bh_varying_theta_error_distribution_{count}.png", dpi = 500)
            count += 1

        # Visualise error distribution
        count = 1
        for theta in theta_range[:3:1]:
            error = all_error[theta]
            root = roots[theta]
            ax = plot_tree(root, particles, error = error, visualise_error=True)
            ax.set_title(f"Spatial error distribution for BH Algorithm for {x_label} = {theta:.3f} and N = {N}")
            plt.tight_layout()
            plt.savefig(f"Figure/bh_varying_theta_spatial_error_distribution_{count}.png", dpi = 500)
            count += 1
    