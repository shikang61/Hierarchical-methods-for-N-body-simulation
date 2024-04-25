import numpy as np
import matplotlib.pyplot as plt

def plot_error_distribution(all_x, all_error, xlabel):
    """
    This function plots the error distribution of the relative error.

    Inputs:
    ---------
    all_x : np.array
        Variable parameter for the simulation
    all_error: list
        The list of every particles relative error for each x
    xlabel: str
        The label for the x-axis

    Returns:
    --------
    ax : matplotlib axis
        return the axis of the plot
    """
    fig, ax = plt.subplots(figsize = (10,6))
    for x, error in zip(all_x, all_error):
        ax.hist(error, bins = 50, alpha = 0.7, color = 'b', edgecolor = 'black', label=f"{xlabel} = {x}")
        ax.set_xlabel("Relative error")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Error distribution of varying {xlabel}")
    return ax
    