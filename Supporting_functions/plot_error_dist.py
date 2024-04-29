import numpy as np
import matplotlib.pyplot as plt

def plot_error_distribution(x, error, xlabel):
    """
    This function plots the error distribution of the relative error.

    Inputs:
    ---------
    x : np.array
        Variable parameter for the simulation
    error: list
        The list of every particles relative error
    xlabel: str
        The label for the x-axis

    Returns:
    --------
    ax : matplotlib axis
        return the axis of the plot
    """
    fig, ax = plt.subplots(figsize = (10,6))
    error = error[np.nonzero(error)]
    bins = np.logspace(np.floor(np.log10(min(error))), np.ceil(np.log10(max(error))), 20)
    ax.hist(error, bins = bins, alpha = 0.7, edgecolor = 'black', label=f"{xlabel} = {x:.3f}")
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel("Relative error")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error distribution of varying {xlabel}")
    ax.legend()
    return ax
    