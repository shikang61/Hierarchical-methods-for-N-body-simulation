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
        If True, x-axis will be log2(x). Default is False.
    log_y : bool
        If True, y-axis will be log2(y). Default is False.
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
                X = np.log2(x)
                ax.set_xlabel(fr"$log_{2}$ {x_label}")
            else:
                X = x
                ax.set_xlabel(f"{x_label}")
            
            # log of y
            if log_y:
                Y = np.log2(y)
                ax.set_ylabel(r"$log_{2}$ (t/s)")
            else:
                Y = y
                ax.set_ylabel("t/s")
            
            # fit a line
            if fit_line:
                fit_x = np.log2(x)
                fit_y = np.log2(y)
                m, c = np.polyfit(fit_x, fit_y, 1)
                ax.plot(fit_x, m*fit_x+ c, label = rf"{key} fit: $t \propto {x_label}^{{{m:.2f}}}$", linestyle = "--")

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
            X = np.log2(x)
            ax.set_xlabel(rf"$log_{2}$ {x_label}")
        else:
            X = x
            ax.set_xlabel(f"{x_label}")
        if log_y:
            Y = np.log2(results)
            ax.set_ylabel(r"$log_{2}$ error")
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    return ax


# def NlogN(N, a, b):
#     return a*N*np.log2(N) + b

# def N2(N, a, b):
#     return a*N**2 + b

# def logN(N, a, b):
#     return a*np.log2(N) + b

# X_fit = np.linspace(min(X), max(X), 100)
# if key == "create_tree":
#     popt, pcov = curve_fit(logN, X, Y)
#     a, b = popt[0], popt[1]
#     ax.scatter(X, Y, label = key)
#     ax.plot(X_fit, logN(X_fit,*popt), label = rf"t = {a:.2f}logN + {b:.2f}", linestyle = "--")
# elif key == "bh_calc":
#     popt, pcov= curve_fit(NlogN, X, Y)
#     a, b = popt[0], popt[1]
#     ax.scatter(X, Y, label = key)
#     ax.plot(X_fit, NlogN(X_fit, *popt), label = rf"t = {a:.2f}NlogN + {b:.2f}", linestyle = "--")
# elif key == "direct_sum":
#     popt, pcov = curve_fit(N2, X, Y)
#     a, b = popt[0], popt[1]
#     ax.scatter(X, Y, label = key)
#     ax.plot(X_fit, N2(X_fit, *popt), label = rf"t = {a:.2f}$N^2$ + {b:.2f}", linestyle = "--")