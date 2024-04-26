# Comparing between Barnes-Hut Algorithm and Fast Multipole Method: Experiment with varying N for a fixed θ, p and levels.

"""
The below code runs the BH and FMM algorithm for different N with fixed (BH) theta = 0.5, (FMM) p = 7 and levels = 5.
We will examine the (1) time taken and (2) error of BH as a function of N.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_FMM_simulation, run_BH_simulation
import matplotlib.pyplot as plt

################## Simulation Parameter ##################
# Fixed Parameters
theta = 0.5             # separation parameter (BH)

p = 7                   # Number of terms in multipole (FMM)
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
max_levels = 4          # Maximum number of levels in the FMM Quadtree
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive

center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
max_n = 1               # Maximum number of particles in a leaf box
seed = 21               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"

# Variable parameter: Number of particles
data_points = 5         # Number of data points
max_N = 500
N_range = np.logspace(2, np.log10(max_N), data_points).astype(int)
x_label = "N"
####################################################################


def fmm_vs_bh():
    # Results variable
    BH_times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []}    # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
    FMM_times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "S2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    BH_max_error = [] # list[float]: A list containing the max error for each N 
    FMM_max_error = []
    BH_all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    FMM_all_error = {} 
    BH_roots = {} # dict[int, Box_bh]: A dictionary containing the root box for each N
    FMM_roots = {}
    BH_FMM_diff_max_error = [] # list[float]: A list containing the max relative error between BH and FMM for each N

    print("\n ************** Start experiment with varying N (number of particles) for Barnes-Hut Algorithm **************\n")

    ################### SIMULATION ###################
    # Print all the simulation parameters
    print(f"""Simulation parameters: \n\t (For BH) theta = {theta}
         (For FMM) p = {p}, boundary = {"wall" if boundary==None else boundary}, maximum levels = {max_levels}
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t Maximum number of particles in leaf box = {max_n} 
         Number of data points = {data_points} \n\t Range of {x_label} = 100 to {max_N} \n""")
    input("Press Enter to begin simulation...")

    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)

        # FMM
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} for Fast Multipole Method ---------\n")        
        phi_fmm = run_FMM_simulation(center, size, p, max_n, boundary, particles, max_levels, N, FMM_tree_type, N, FMM_times, FMM_max_error, FMM_all_error, FMM_roots)
        
        # BH
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} for Barnes-Hut Method ---------\n")
        phi_bh = run_BH_simulation(center, size, max_n, theta, particles, N, N, BH_times, BH_max_error, BH_all_error, BH_roots)

        BH_FMM_diff_error = calculate_error(phi_bh, phi_fmm)
        BH_FMM_diff_max_error.append(np.max(BH_FMM_diff_error))

    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if input("Do you want to save the data? (y/n): ") == "y":
        np.save(f"Data/FMM_vs_BH_time_FMM_{p}_{max_levels}_{FMM_tree_type}_{max_N}", FMM_times)
        np.save(f"Data/FMM_vs_BH_error_FMM_{p}_{max_levels}_{FMM_tree_type}_{max_N}", FMM_all_error)
        np.save(f"Data/FMM_vs_BH_time_BH_{theta}_{max_N}", BH_times)
        np.save(f"Data/FMM_vs_BH_error_BH_{theta}_{max_N}", BH_all_error)
        print("--- Data saved successfully ---\n")

    ################### GENERATING PLOTS ###################
    plt.close()
    # Plot time against N
    if input(f"Do you want to plot the results for time vs {x_label}? (y/n): ") == "y":
        all_times = {**FMM_times, **BH_times}
        keys = ["fmm_calc", "fmm_direct_sum" , "bh_calc"]
        all_times = {key: np.array(all_times[key]) for key in keys}
        
        # Renaming key from "fmm_direct_sum" to "direct_sum"
        all_times["direct_sum"] = all_times["fmm_direct_sum"]
        del all_times["fmm_direct_sum"]

        # log2(t) vs log2(N)
        ax = plot_results(N_range, all_times, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
        ax.set_title(r"$log_{2}$ t vs $log_{2}$ N for both FMM and BH for " + rf"$\theta$ = {theta}, p = {p}, levels = {max_levels}, tree type = {FMM_tree_type}") 
        plt.savefig("Figure/bh_vs_fmm_varying_N_logt_vs_logN.jpg")    

    # Plot max_error against N, error vs log2(N)
    if input(f"Do you want to plot the results for error vs {x_label}? (y/n): ") == "y":
        # ax = plot_results(N_range, BH_FMM_diff_max_error, x_label, plot_style = "line", log_x = True, log_y = True, fit_line = False, error_label="_FMM_BH")
        ax = plot_results(N_range, BH_max_error, x_label, plot_style = "line", log_x = True, log_y = True, fit_line = False, ax=ax, error_label="_BH_direct")
        ax = plot_results(N_range, FMM_max_error, x_label, plot_style = "line", log_x = True, log_y = True, fit_line = False, ax=ax, error_label="_FMM_direct")
        ax.set_title(r"Max relative error between FMM and BH vs $log_{2}$ N for " + rf"$\theta$ = {theta}, p = {p}, levels = {max_levels}, tree type = {FMM_tree_type}")
        plt.savefig("Figure/bh_vs_fmm_varying_N_error_vs_logN.png")    

    plt.show()