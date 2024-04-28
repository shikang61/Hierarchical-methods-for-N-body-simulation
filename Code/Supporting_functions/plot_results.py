import matplotlib.pyplot as plt
import numpy as np

def plot_results(x, results, x_label, plot_style, log_x = False, log_y = False, fit_line = False, ax = None, error_label = ""):
    """
    This function plot the results for the simulation

    Inputs:
    -------
    x : np.array
        Variable parameter for the simulation
    results : dict or list
        Results can be time or error.

        1) If time, then type(results) = dict[str, list[float]]
        For BH: results = {"create_tree": List[float], "bh_calc": List[float], "direct_sum": List[float]}
        For FMM: results = {"create_tree": List[float], "S2M_time": List[float], "M2M_time": List[float], 
                            "M2L_time": List[float], "L2L_time": List[float], "L2P_time": List[float], 
                            "P2P_time": List[float], "total_time": List(float), "direct_sum": List[float]}
        
        2) If error, then type(results) = list[float]

    x_label : str
        Label for the x-axis
    plot_style: str
        Type of plot: "scatter" or "line"
    log_x : bool 
        If True, x-axis will be log10(x). Default is False.
    log_y : bool
        If True, y-axis will be log10(y). Default is False.
    fit_line : bool
        If True, a line will be fitted to the data. Default is False.

    
    Returns:
    --------
    ax : matplotlib axis
        return the axis of the plot
    """
    if ax == None:
        fig, ax = plt.subplots(figsize = (10,6))
    if type(results) == dict:
        for key in results.keys():
            y = np.array(results[key])

            # log of x
            if log_x:
                X = np.log10(x)
                ax.set_xlabel(r"$log_{10}$ "+f"{x_label}")
            else:
                X = x
                ax.set_xlabel(f"{x_label}")
            
            # log of y
            if log_y:
                Y = np.log10(y)
                ax.set_ylabel(r"$log_{10}$ (t/s)")
            else:
                Y = y
                ax.set_ylabel("t/s")
            
            # fit a line
            if fit_line:
                try:
                    fit_x = np.log10(x)
                    fit_y = np.log10(y)
                    m, c = np.polyfit(fit_x, fit_y, 1)
                    fit_x = np.linspace(min(fit_x), max(fit_x), 500)
                    ax.plot(fit_x, m*fit_x+ c, label = rf"{key} fit: $t \propto {x_label}^{{{m:.2f}}}$", linestyle = "--")
                except:
                    pass

            # Plot style "line" or "scatter". Do not label the points if fit_line is True
            if plot_style == "line":
                if fit_line:
                    ax.plot(X, Y)
                else:
                    ax.plot(X, Y, label = key)
            elif plot_style == "scatter":
                if fit_line:
                    ax.scatter(X, Y, s = 10)
                else:
                    ax.scatter(X, Y, label = key, s= 10) 

    elif type(results) == list:
        # log of x
        if log_x:
            X = np.log10(x)
            ax.set_xlabel(r"$log_{10}$ " + f"{x_label}")
        else:
            X = x
            ax.set_xlabel(f"{x_label}")
        if log_y:
            Y = np.log10(results)
            ax.set_ylabel(r"$log_{10}$ error")
        else:
            Y = results
            ax.set_ylabel("Max relative error")
        
        # plot style
        if plot_style == "line":
            ax.plot(X, Y, label = "Max relative error" + error_label)
        elif plot_style == "scatter":
            ax.scatter(X, Y, label = "Max relative error" + error_label, s=10)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2) # 4
    return ax

