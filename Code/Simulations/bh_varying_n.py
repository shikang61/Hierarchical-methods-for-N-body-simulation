# Barnes-Hut Algorithm: Experiment with varying N (number of particles)
"""
The below code runs the BH algorithm for different N with fixed theta = 0.5.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from Barnes_Hut_Algo import BH_build_tree, BH_calculate_potential_all, BH_potential_direct_sum
from Class import Box_bh
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
N_range = np.logspace(2,2.5,data_points).astype(int) #N = [  100,   166,   278,   464,   774,  1291,  2154,  3593,  5994, 10000] for 10 data points
x_label = "N"
####################################################################

# Results variable
times = {"create_tree": [], "bh_calc": [], "direct_sum": []}    # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
max_error = [] # list[float]: A list containing the max error for each N
all_error = [] # list[list[float]]: A list containing the error of every particle for each N

def bh_varying_n():
    print("\n ************** Start experiment with varying N (number of particles) for Barnes-Hut Algorithm **************")
    for i, N in enumerate(N_range):
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
        times["direct_sum"].append(direct_time)
        phi_direct = get_potential(particles)
        print(f"Time taken to calculate the potential directly: {direct_time:.4f} seconds")

        # Error
        error = calculate_error(phi_bh, phi_direct)
        max_error.append(np.max(error))
        all_error.append(error)
        print(f"Max relative error: {np.max(error)*100:.4f}%")
        
    print("\n ************** Simulation Done ************** \n")

    if input("Do you want to save the data? (y/n): ") == "y":
        np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_time", times)
        np.save(f"Data/BH_varying_N_{max(N_range)}_{theta}_error", all_error)
        print("--- Data saved successfully ---\n")

    plt.close()

    # Plot time against N
    if input(f"Do you want to plot the results for time vs thata? (y/n): ") == "y":
        # log2(t) vs log2(N)
        ax = plot_results(N_range, times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
        ax.set_title(r"BH: $log_{2}$ t vs $log_{2}$ N for " + r"$\theta$ =" f"{theta}")

    # Plot max_error against N
    if input(f"Do you want to plot the results for error vs theta? (y/n): ") == "y":
        # error vs log2(N)
        ax = plot_results(N_range, max_error, x_label, plot_style = "line", log_x = True, log_y = False, fit_line = False)
        ax.set_title(r"BH: max_error vs $log_{2}$ N for " + r"$\theta$ =" f"{theta}")
    
    # Visualise the Quadtree
    if input("Do you want to visualise the Quadtree? (y/n): ") == "y":
        ax = plot_tree(root, particles)
        ax.set_title("Quadtree for BH Algorithm")
    
    plt.show()


    




    
