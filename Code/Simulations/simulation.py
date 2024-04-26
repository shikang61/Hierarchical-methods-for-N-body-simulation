from FMM_Algo import FMM_build_fixed_tree, FMM_build_adaptive_tree, FMM_calculate_potential_all, FMM_potential_direct_sum
from Barnes_Hut_Algo import BH_build_tree, BH_calculate_potential_all, BH_potential_direct_sum
from Supporting_functions import *
from Class import Box_bh, Box_fmm
import numpy as np

def run_FMM_simulation(center, 
                       size, 
                       p, 
                       max_n,
                       boundary, 
                       particles, 
                       max_levels, 
                       N, 
                       tree_type,
                       variable,
                       FMM_times, 
                       FMM_max_error, 
                       FMM_all_error,
                       FMM_roots):
        """
        Function to run the FMM simulation
        """
        root = Box_fmm(coords = center, 
                    size = size, 
                    p = p,
                    max_n = max_n,
                    boundary = boundary)
        
        FMM_roots[variable] = root

        # Build the tree
        if tree_type == "adaptive":
            tree_time = FMM_build_adaptive_tree(root, particles)
        elif tree_type == "fixed":
            tree_time = FMM_build_fixed_tree(root, particles, max_levels)

        FMM_times["fmm_create_tree"].append(tree_time)
        print(f"Time taken to build the {tree_type} tree: {tree_time:.4f} seconds")

        # Calculate the potential
        potential_time = FMM_calculate_potential_all(root)
        keys = list(potential_time.keys())
        for key in keys:
            FMM_times[key].append(potential_time[key])

        phi_fmm = get_potential(particles)
        print(f"Time taken to calculate the potential: {round(potential_time['fmm_calc'],4)} seconds")
        print(f"Total time taken for the FMM algorithm: {round(tree_time + potential_time['fmm_calc'], 4)} seconds")
    
        # Direct sum
        direct_time = FMM_potential_direct_sum(particles, tqdm_bar=True)
        FMM_times["fmm_direct_sum"].append(direct_time)
        phi_direct = get_potential(particles)
        print(f"Time taken to calculate the potential directly: {direct_time:.4f} seconds")

        # Error
        error = calculate_error(phi_fmm, phi_direct)
        FMM_max_error.append(np.max(error))
        FMM_all_error[variable] = error
        print(f"Max relative error: {np.max(error)*100:.4f}%")

        return phi_fmm


def run_BH_simulation(center, 
                      size, 
                      max_n, 
                      theta, 
                      particles, 
                      N, 
                      variable, 
                      BH_times, 
                      BH_max_error, 
                      BH_all_error, 
                      roots):
    """
    Function to run the BH simulation
    """
    root = Box_bh(coords = center, 
                    size = size, 
                    max_n = max_n,
                    theta = theta)
    
    roots[variable] = root
    # Build the tree
    tree_time = BH_build_tree(root, particles)
    BH_times["bh_create_tree"].append(tree_time)
    print(f"Time taken to build the tree: {tree_time:.4f} seconds")

    # Calculate the potential
    potential_time = BH_calculate_potential_all(particles, root)
    BH_times["bh_calc"].append((potential_time))
    phi_bh = get_potential(particles)
    print(f"Time taken to calculate the potential: {potential_time:.4f} seconds")
    print(f"Total time taken for the Barnes-Hut algorithm: {tree_time + potential_time:.4f} seconds")

    # Direct sum
    direct_time = BH_potential_direct_sum(particles, tqdm_bar=True)
    BH_times["bh_direct_sum"].append(direct_time)
    phi_direct = get_potential(particles)
    print(f"Time taken to calculate the potential directly: {direct_time:.4f} seconds")

    # Error
    error = calculate_error(phi_bh, phi_direct)
    BH_max_error.append(np.max(error))
    BH_all_error[variable] = error
    print(f"Max relative error: {np.max(error)*100:.4f}%")

    return phi_bh