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
N = 1000             # Number of particles
m = 1               # Maximum number of particles in a m box

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
direct_sum = True      # Option to calculate the direct sum

# Variable parameter: separation parameter θ 
data_points = 35         # Number of data points
min_theta = 0.01
max_theta = 19
theta_range = np.linspace(min_theta, max_theta, data_points) 
x_label = r"$\theta$"
####################################################################

def bh_varying_theta():

    # Results variable
    times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []} # dict[str, list[float]]: A dictionary containing the time taken for each step for each θ
    rel_error = []  # list[float]: A list containing the relative error for each θ
    all_error = {}  # list[list[float]]: A list containing the error of every particle for each θ
    roots = {}

    print("\n ************** Start experiment with varying θ (separation parameter) for Barnes-Hut Algorithm **************\n")
    print(f"""Simulation parameters:
         Number of particles = {N}
         Size of the simulation area = {size} 
         Seed = {seed}
         Distribution of particles = {distribution}
         Maximum number of particles in m box = {m} 
         Number of data points = {data_points}, ange of {x_label} = {min_theta} to {max_theta} \n""")
    
    start_time = time.perf_counter()
    for i, theta in enumerate(theta_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{data_points}) theta = {theta} ---------\n")
        run_BH_simulation(center, size, m, theta, particles, N, theta, times, rel_error, all_error, roots, direct_sum)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    print("\n ************** Simulation Done ************** \n")
    
    # # Option to save simulation data
    if save_data:
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_time", times)
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    # log(time) against theta
    ax = plot_results(theta_range, times, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)  
    ax.set_title(r"BH: lg t "+ f"vs {x_label} for " + r"N = " f"{N}")
    plt.savefig("Figure/bh_varying_theta_logt_vs_theta.png", dpi = 500)

    if direct_sum:
        # Plot rel_error against theta
        ax = plot_results(theta_range, rel_error, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(f"BH: relative error vs {x_label} for N =" f"{N}")    
        plt.savefig("Figure/bh_varying_theta_relerr_vs_theta.png", dpi = 500)

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
    