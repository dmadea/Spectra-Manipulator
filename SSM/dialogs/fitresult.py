import numpy as np
from lmfit import fit_report, report_fit, report_ci, conf_interval, conf_interval2d

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ..user_namespace import plot_fit


class FitResult(object):
    """Helper class that stores the results after fitting and provide additional functions such as printing
    or plotting confidence intervals.

    Attributes
    ----------

    result : :class:`MinimizerResult`
        Result from fitting as obtained by LMFIT.
    minimizer : :class:`Minimizer`
        Minimizer class used for fitting.
    params : :class:`Parameters`
        Variable that stored parameters as :class:`MinimizerResult` class
    values_errors : numpy.ndarray
        2D numpy array of fitted paramater values (first column) and corresponding errors (second column)
    model : :class:`_Model`
        Model used for fitting.
    """

    def __init__(self, minimizer_result, minimizer, values_errors, current_model,
                 data_item=None, fit_item=None, residuals_item=None):
        self.result = minimizer_result
        self.minimizer = minimizer
        self.params = minimizer_result.params
        self.values_errors = values_errors
        self.model = current_model

        self.data_item = data_item
        self.fit_item = fit_item
        self.residuals_item = residuals_item

    def report(self, print=True):
        """Prints a fit report if print == True, otherwise returns fit report as text.

        Parameters
        ----------
        print : bool
            If True, fit report is printed.
        """
        return report_fit(self.result) if print else fit_report(self.result)

    # https://lmfit.github.io/lmfit-py/confidence.html
    def confidence_intervals(self, p_names=None, sigmas=(1, 2, 3)):
        """Prints a confidence intervals.

        Parameters
        ----------
        p_names : {list, None}, optional
            Names of the parameters for which the confidence intervals are calculated. If None (default),
            the confidence intervals are calculated for every parameter.
        sigmas : {list, tuple}, optional
            The sigma-levels to find (default is [1, 2, 3]). See Note below.

        Note
        ----
        The values for sigma are taken as the number of standard deviations for a normal distribution
        and converted to probabilities. That is, the default sigma=[1, 2, 3] will use probabilities of
        0.6827, 0.9545, and 0.9973. If any of the sigma values is less than 1, that will be interpreted
        as a probability. That is, a value of 1 and 0.6827 will give the same results, within precision.

        """
        ci = conf_interval(self.minimizer, self.result, p_names=p_names, sigmas=sigmas)
        report_ci(ci)

    # https://lmfit.github.io/lmfit-py/confidence.html
    def confidence_interval2D(self, x_name, y_name, nx=10, ny=10):
        """Draws a 2D confidence intervals using matplotlib.

        Parameters
        ----------
        x_name : str
            Name of the variable that will be on the x axis.
        y_name : str
            Name of the variable that will be on the y axis.
        nx : int, optional
            Number of points in the x direction, default 10, the higher the value, better resolution, but slower.
        ny : int, optional
            Number of points in the y direction, default 10, the higher the value, better resolution, but slower.
        """
        cx, cy, grid = conf_interval2d(self.minimizer, self.result, x_name, y_name, nx, ny)
        plt.contourf(cx, cy, grid, np.linspace(0, 1, 21))
        plt.xlabel(x_name)
        plt.colorbar()
        plt.ylabel(y_name)
        plt.show()

    def plot(self, symlog=False, linscale=1, linthresh=100,
                  lw_data=0.5, lw_fit=1.5, fig_size=(5, 4), y_label='$\\Delta$A', x_label='Time / $\\mu$s',
                  x_lim=(None, None), x_major_formatter=ScalarFormatter(), x_major_locator=None, y_major_locator=None,
             data_color='red', t_mul_factor=1, y_lim=(None, None), filepath=None, dpi=500, transparent=False):

        plot_fit(self.data_item, self.fit_item, self.residuals_item, symlog=symlog, linscale=linscale,
                  linthresh=linthresh, lw_data=lw_data, lw_fit=lw_fit, fig_size=fig_size, y_label=y_label,
                  x_label=x_label, x_lim=x_lim, t_mul_factor=t_mul_factor, y_lim=y_lim, filepath=filepath,
                  dpi=dpi, transparent=transparent, x_major_formatter=x_major_formatter,
                  x_major_locator=x_major_locator, y_major_locator=y_major_locator, data_color=data_color)
