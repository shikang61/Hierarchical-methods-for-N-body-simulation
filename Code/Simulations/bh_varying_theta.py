# Barnes-Hut Algorithm: Experiment with varying θ (separation parameter for Barnes-Hut Algorithm)
"""
The below code runs the BH algorithm for different θ with fixed N = 500.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from Barnes_Hut_Algo import BH_build_tree, BH_calculate_potential_all, BH_potential_direct_sum
from Class import Box_bh
import matplotlib.pyplot as plt

################## Simulation Parameter ##################
# Fixed Parameters
N = 100                 # Number of particles
center = (0, 0)         # Center of the simulation area
size = 1024              # Size of the simulation area
max_n = 1               # Maximum number of particles in a leaf box
seed = 21               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"

# Variable parameter: separation parameter θ 
data_points = 30         # Number of data points
theta_range = np.linspace(0.001, 15, data_points) 
x_label = r"$\theta$"
####################################################################

# Results variable
times = {"create_tree": [], "bh_calc": [], "direct_sum": []} # dict[str, list[float]]: A dictionary containing the time taken for each step for each θ
max_error = []  # list[float]: A list containing the max error for each θ
all_error = []  # list[list[float]]: A list containing the error of every particle for each θ

def bh_varying_theta():
    print("\n ************** Start experiment with varying θ (separation parameter) for Barnes-Hut Algorithm **************")
    
    for i, theta in enumerate(theta_range):
        particles = generate_particles(N, size, seed, distribution)
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} and theta = {theta} ---------\n")
        root = Box_bh(coords = center, 
                    size = size, 
                    max_n = max_n,
                    theta = theta)
        # Build the tree
        tree_time = BH_build_tree(root, particles)
        times["create_tree"].append(tree_time)
        print(f"Time taken to build the tree: {tree_time:.4f} seconds")

        # Calculate the potential
        potential_time = BH_calculate_potential_all(particles, root)
        times["bh_calc"].append((potential_time))
        phi_bh = get_potential(particles)
        print(f"Time taken to calculate the potential: {potential_time:.4f} seconds")
        print(f"Total time taken for the Barnes-Hut algorithm: {tree_time + potential_time:.4f} seconds")
    
        # Direct sum
        direct_time = BH_potential_direct_sum(particles)
        times["direct_sum"].append((direct_time))
        phi_direct = get_potential(particles)
        print(f"Time taken to calculate the potential directly: {direct_time:.4f} seconds")

        # Error
        error = calculate_error(phi_bh, phi_direct)
        max_error.append(np.max(error))
        all_error.append(error)
        print(f"Max relative error: {np.max(error)*100:.4f}%")
    
    print("\n ************** Simulation Done ************** \n")

    if input("Do you want to save the data? (y/n): ") == "y":
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_time", times)
        np.save(f"Data/BH_varying_theta_{max(theta_range)}_{N}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()

    # Plot time against theta
    if input(rf"Do you want to plot the results for time vs {x_label}? (y/n): ") == "y":
        ax = plot_results(theta_range, times, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)  
        ax.set_title(r"BH: $log_2$ t vs $\theta$ for " + r"N =" f"{N}")

    # Plot max_error against theta
    if input(rf"Do you want to plot the results for error vs {x_label}? (y/n): ") == "y":
        ax = plot_results(theta_range, max_error, x_label, plot_style = "line", log_x = False, log_y = False, fit_line = False)
        ax.set_title(r"BH: max_error vs $\theta$ for " + r"N =" f"{N}")    

    # Visualise the Quadtree
    if input("Do you want to visualise the Quadtree? (y/n): ") == "y":
        ax = plot_tree(root, particles)
        ax.set_title("Quadtree for BH Algorithm")

    if input("Do you want to plot the error distribution? (y/n): ") == "y":
        ax = plot_error_distribution(theta_range, all_error, x_label)
        ax.set_title("Error distribution for BH Algorithm")
    
    plt.show()


    




    
