# Barnes-Hut Algorithm: Experiment with varying θ (separation parameter for Barnes-Hut Algorithm)
"""
The below code runs the BH algorithm for different θ with fixed N = 500.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_BH_simulation
from Class import Box_bh
import matplotlib.pyplot as plt

################## Simulation Parameter ##################
# Fixed Parameters
N = 200                 # Number of particles
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
max_n = 1               # Maximum number of particles in a leaf box
seed = 21               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"

# Variable parameter: separation parameter θ 
data_points = 30         # Number of data points
max_theta = 10
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
    
    ################### SIMULATION ###################
    # Print all the simulation parameters
    print(f"""Simulation parameters: \n\t Number of particles = {N} \n\t Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {max_n} 
         Number of data points = {data_points} \n\t Range of {x_label} = 0.01 to {max_theta} \n""")
    input("Press Enter to begin simulation...")
    
    for i, theta in enumerate(theta_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{data_points}) theta = {theta} ---------\n")
        run_BH_simulation(center, size, max_n, theta, particles, N, theta, times, max_error, all_error, roots)
    
    print("\n ************** Simulation Done ************** \n")
    
    # Option to save simulation data
    if input("Do you want to save the data? (y/n): ") == "y":
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_time", times)
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_error", all_error)
        print("--- Data saved successfully ---\n")

    ################### GENERATING PLOTS ###################
    plt.close()
    # Plot time against theta
    if input(f"Do you want to plot the results for time vs {x_label}? (y/n): ") == "y":
        # log2(t) vs θ
        ax = plot_results(theta_range, times, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)  
        ax.set_title(fr"BH: $log_2$ t vs {x_label} for " + r"N =" f"{N}")
        plt.savefig("Figure/bh_varying_theta_logt_vs_theta.png")
        
        # t vs θ
        ax = plot_results(theta_range, times, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)  
        ax.set_title(fr"BH: t vs {x_label} for " + r"N =" f"{N}")
        plt.savefig("Figure/bh_varying_theta_t_vs_theta.png")

    # Plot max_error against theta
    if input(f"Do you want to plot the results for error vs {x_label}? (y/n): ") == "y":
        ax = plot_results(theta_range, max_error, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(fr"BH: max error vs {x_label} for " + r"N =" f"{N}")    
        plt.savefig("Figure/bh_varying_theta_maxerr_vs_theta.png")

    # Visualise the Quadtree
    if input("Do you want to visualise the Quadtree? (y/n): ") == "y":
        ax = plot_tree(list(roots.values())[0], particles, visualise_interaction=True)
        ax.set_title(fr"Quadtree for BH Algorithm for {x_label} = 0.5 and N = {N}")
        plt.savefig("Figure/bh_varying_theta_quadtree_interactions.png")

    # Plot error distribution
    if input("Do you want to plot the error distribution? (y/n): ") == "y":
        for x, error in zip(theta_range[:4:1], list(all_error.values())[:4:1]): 
            ax = plot_error_distribution(x, error, x_label)
            ax.set_title(f"Error distribution for BH Algorithm for N = {N}")
            plt.savefig("Figure/bh_varying_theta_error_distribution.png")
             
    # Visualise the error distribution
    if input("Do you want to visualise the spatial error distribution? (y/n): ") == "y":
        print(theta_range[:10])
        while True:
            try:
                theta_idx = int(input("Enter the index of the value of theta you want to visualise: "))
                theta = theta_range[theta_idx]
                break
            except:
                print("invalid input")
                pass
        
        
        error = all_error[theta]
        root = roots[theta]
        ax = plot_tree(root, particles, error = error, visualise_error=True)
        ax.set_title(f"Spatial error distribution for BH Algorithm for {x_label} = {theta:.3f} and N = {N}")
        plt.tight_layout()
        plt.savefig("Figure/bh_varying_theta_spatial_error_distribution.png")
    
    plt.show()