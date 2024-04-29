"""
The below code runs both the BH and FMM algorithm for different N with fixed (BH) theta = 0.5, (FMM) p = 6 and dynamic levels.
It will compare the run time of BH and FMM as well as their relative error.
"""
import numpy as np
from Supporting_functions import *
from .simulation import run_FMM_simulation, run_BH_simulation
import matplotlib.pyplot as plt
import time
from FMM_Algo import set_max_levels

################## Simulation Parameter ##################
# Fixed Parameters for the algorithm
theta = 0.5             # separation parameter (BH)
p = 6                   # Number of terms in multipole (FMM)
boundary = None         # Boundary condition of the simulation: "None" for wall boundary condition, "periodic" for periodic boundary condition (FMM)
FMM_tree_type = "fixed" # Type of FMM tree: "fixed" or "adaptive
bh_m = 1            # Maximum number of particles in a m box (for BH)
fmm_m = 10           # Maximum number of particles in a m box (for FMM)

# Fixed parameters for the environment
center = (0, 0)         # Center of the simulation area
size = 1024             # Size of the simulation area
seed = 25               # Seed for random number generator
distribution = "random" # Distribution of the particles: "random" or "triangular"
save_data = False       # Option to save the data
# direct_sum = True

# Variable parameter: Number of particles
data_points = 20         # Number of data points
max_N = 10000
N_range = np.logspace(2, np.log10(max_N), data_points).astype(int) #[100, 143, 206, 297, 428, 615, 885, 1274, 1832, 2636, 3792, 5455, 7847, 11288, 16237, 23357, 33598, 48329, 69519, 100000]
x_label = "N"
####################################################################

def fmm_vs_bh():
    # Results variable
    BH_times = {"bh_create_tree": [], "bh_calc": [], "bh_direct_sum": []}    # dict[str, list[float]]: A dictionary containing the time taken for each step for each N
    FMM_times = {"fmm_create_tree": [], "fmm_calc": [], "fmm_direct_sum": [], "P2M_time": [], "M2M_time": [], "M2L_time": [], "L2L_time": [], "L2P_time": [], "P2P_time": []}   
    BH_rel_error = [] # list[float]: A list containing the relative error for each N 
    FMM_rel_error = []
    BH_all_error = {} # list[list[float]]: A list containing the error of every particle for each N
    FMM_all_error = {} 
    BH_roots = {} # dict[int, Box_bh]: A dictionary containing the root box for each N
    FMM_roots = {}
    BH_FMM_diff_rel_error = [] # list[float]: A list containing the max relative error between BH and FMM for each N

    print("\n ************** Start experiment with varying N (number of particles) for BH Algorithm and FMM Algorithm **************\n")
    print(f"""Simulation parameters: \n\t (For BH) theta = {theta}
         (For FMM) p = {p}, boundary = {"wall" if boundary==None else boundary}, maximum levels = log4(N)
         (For FMM) Maximum number of particles in m box = {fmm_m} 
         Size of the simulation area = {size} 
         Seed = {seed} \n\t Distribution of particles = {distribution} \n\t 
         Number of data points = {data_points} \n\t Range of {x_label} = 100 to {max_N} \n""")

    start_time = time.perf_counter()
    for i, N in enumerate(N_range):
        particles = generate_particles(N, size, seed, distribution)
        max_levels = set_max_levels(N, fmm_m) # dynamically set max_levels based on N
        # FMM
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} for Fast Multipole Method ---------\n")        
        phi_fmm = run_FMM_simulation(center, size, p, fmm_m, boundary, particles, max_levels, N, FMM_tree_type, N, FMM_times, FMM_rel_error, FMM_all_error, FMM_roots, direct_sum=True)
        
        # BH
        print(f"\n--------- ({i+1}/{data_points}) Number of particles = {N} for Barnes-Hut Method ---------\n")
        phi_bh = run_BH_simulation(center, size, bh_m, theta, particles, N, N, BH_times, BH_rel_error, BH_all_error, BH_roots, direct_sum=False)

        BH_FMM_diff_error = relative_error(phi_bh, phi_fmm)
        BH_FMM_diff_rel_error.append(np.max(BH_FMM_diff_error))
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    print("\n ************** Simulation Done ************** \n")

    # Option to save simulation data
    if save_data:
        np.save(f"Data/FMM_vs_BH_time_FMM_{p}_{max_levels}_{FMM_tree_type}_{max_N}", FMM_times)
        np.save(f"Data/FMM_vs_BH_error_FMM_{p}_{max_levels}_{FMM_tree_type}_{max_N}", FMM_all_error)
        np.save(f"Data/FMM_vs_BH_time_BH_{theta}_{max_N}", BH_times)
        np.save(f"Data/FMM_vs_BH_error_BH_{theta}_{max_N}", BH_all_error)
        print("--- Data saved successfully ---\n")

    plt.close()
    all_times = {**FMM_times, **BH_times}
    
    # log(t) vs N - calc
    keys = ["fmm_calc", "bh_calc", "fmm_direct_sum"]
    calc_time = {key: np.array(all_times[key]) for key in keys}
    ax = plot_results(N_range, calc_time, x_label, plot_style = "line", log_x = False, log_y = True, fit_line = False)
    ax.set_title(r"lg t vs N for both FMM and BH for " + rf"$\theta$ = {theta}, p = {p}, " + r"$m_{fmm}$ " + f"= {fmm_m}") 
    plt.savefig("Figure/bh_vs_fmm_varying_N_logt_vs_logN_calc_1.jpg", dpi = 500)   

    # log(t) vs log(N) - calc
    keys = ["fmm_calc", "bh_calc", "fmm_direct_sum"]
    calc_time = {key: np.array(all_times[key]) for key in keys}
    ax = plot_results(N_range, calc_time, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"lg t vs lg N for both FMM and BH for " + rf"$\theta$ = {theta}, p = {p}, " + r"$m_{fmm}$ " + f"= {fmm_m}") 
    plt.savefig("Figure/bh_vs_fmm_varying_N_logt_vs_logN_calc_2.jpg", dpi = 500) 

    # log(t) vs log(N) - tree
    keys = ["fmm_create_tree", "bh_create_tree"]
    tree_time = {key: np.array(all_times[key]) for key in keys}
    ax = plot_results(N_range, tree_time, x_label, plot_style = "scatter", log_x = True, log_y = True, fit_line = True)
    ax.set_title(r"lg t vs lg N for both FMM and BH for " + rf"$\theta$ = {theta}, p = {p}, "+ r"$m_{fmm}$ " + f"= {fmm_m}") 
    plt.savefig("Figure/bh_vs_fmm_varying_N_logt_vs_logN_tree.jpg", dpi = 500)   

    # Plot log(relative error) against logN
    ax = plot_results(N_range, BH_FMM_diff_rel_error, x_label, plot_style = "line", log_x = True, log_y = True, fit_line = False, error_label=" FMM_BH")
    ax.set_title(r"Max relative error between FMM and BH vs lg N for " + rf"$\theta$ = {theta}, p = {p}, "+ r"$m_{fmm}$ " + f"= {fmm_m}")
    plt.savefig("Figure/bh_vs_fmm_varying_N_error_vs_logN.png", dpi = 500)    
