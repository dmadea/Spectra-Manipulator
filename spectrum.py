import numpy as np
import os
# from sys import platform

from copy import deepcopy
import math

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq

# import lmfit
import csv
import io

from numba import jit


# from typing import Union, List, Iterable


class CalcMode:
    add_sub = 0
    mul_div = 1


class Spectrum(object):

    def __init__(self, data, filepath=None, name='', group_name='', color=None, line_width=None, line_alpha=255,
                 line_type=None, symbol=None, symbol_brush=None, sym_brush_alpha=255, symbol_fill=None,
                 sym_fill_alpha=255, symbol_size=8, plot_legend=True):
        """
        Class that holds the spectrum object as a 2D array (dimensions n x 2) where n is a number
        of points in spectrum and includes various functions used for data manipulation with them.
        Spectrum is stored in variable *data* as a numpy ndarray.

        Parameters
        ----------
        data : {numpy.ndarray, None}
            This argument represents data, this must be 2D numpy array.
            First column must contain x values and second y values. Data are automatically sorted according to
            first column.
        filepath : str, optional
            The path to the file the spectrum was imported from.
        name : str, optional
            Name of the spectrum.
        group_name : str, optional
            Group name of the spectrum (if inside group).
        color : {str, tuple}, optional
            Color of the spectrum, use string for common colors (eg. 'black', 'blue', 'red', ...)
            or tuple - red, green, blue, alpha components from 0 - 255, eg. (255, 0, 0, 255).
            If None, user defined color scheme will be used.
        line_width : {int, float}, optional
            Sets the line width of the plotted line. If None, user defined color scheme will be used.
        line_type : int 0-6, optional
            Sets the line type of the plotted line. See https://doc.qt.io/archives/qt-4.8/qt.html#PenStyle-enum
            for line types. If None, user defined color scheme will be used.
        plot_legend : bool
            Sets, if legend will be plotted for this spectrum.

        Attributes
        ----------
        data : numpy.ndarray
            A raw data of spectrum object, dimensions are (n x 2) where n is a number of points in spectrum.
            First column represents *x* values, second column *y* values.
        name : str
            The name of the spectrum as is displayed in Tree Widget.
        filepath : {str, None}
            If the spectrum was imported from file, this variable stored the path to the file it was imported from.
        color : {str, tuple, None}
            Color of the spectrum.
        line_width : {int, float, None}
            Line width of the plotted line.
        line_type : {int 0-6, None}
            Line type.
        plot_legend : bool
            If True, name of this spectrum will be added to legend. If False, legend will not contain this spectrum.
        """
        # sort according to first column,numpy matrix, 1. column wavelength, 2. column absorbance
        self.data = np.asarray(data[data[:, 0].argsort()], dtype=np.float64) if data is not None else None
        self.filepath = filepath
        self.name = name
        self.group_name = '' if group_name is None else group_name

        self.color = color  # line color
        self.line_width = line_width
        self.line_type = line_type
        self.plot_legend = plot_legend

        self.symbol = symbol
        self.symbol_brush = symbol_brush
        self.symbol_fill = symbol_fill
        self.symbol_size = symbol_size

        self.line_alpha = line_alpha
        self.sym_brush_alpha = sym_brush_alpha
        self.sym_fill_alpha = sym_fill_alpha

    @classmethod
    def from_xy_values(cls, x_values, y_values, name='', group_name='', color=None, line_width=None, line_alpha=255,
                       line_type=None, symbol=None, symbol_brush=None, sym_brush_alpha=255, symbol_fill=None,
                       sym_fill_alpha=255, symbol_size=8, plot_legend=True):
        """
        Creates the Spectrum object from separate x and y data variables. The dimensions of x_values
        and y_values must be the same and contain numbers. The other parameters are the same
        as for __init__ method.

        Parameters
        ----------
        x_values : iterable, eg. list, tuple, ndarray
            Iterable that represents x values.
        y_values : iterable, eg. list, tuple, ndarray
            Iterable that represents y values.

        Returns
        -------
        out : :class:`Spectrum`
            Spectrum object.


        Raises
        ------
        ValueError
            If *x_values* or *y_values* have not the same dimension or do not contain numbers.
        """
        try:
            if len(x_values) != len(y_values):
                raise ValueError("Length of x_values and y_values must match.")

            x_data = np.asarray(x_values, dtype=np.float64)
            y_data = np.asarray(y_values, dtype=np.float64)
        except ValueError:
            raise
        except:
            raise ValueError(
                "Argument error, x_values and y_values must be iterables and contain numbers.")

        data = np.vstack((x_data, y_data)).T

        return cls(data, name=name, group_name=group_name, color=color, line_width=line_width, line_alpha=line_alpha,
                   line_type=line_type, symbol=symbol, symbol_brush=symbol_brush, sym_brush_alpha=sym_brush_alpha,
                   symbol_fill=symbol_fill, sym_fill_alpha=sym_fill_alpha, symbol_size=symbol_size,
                   plot_legend=plot_legend)

    def add_to_list(self):
        """Adds this spectrum to the Tree Widget. Only works in SSM's Console Widget."""
        try:
            from user_namespace import add_to_list
            add_to_list(self)
        except ImportError:
            pass

    def _redraw_all_spectra(self):
        pass

    def _update_view(self):
        pass

    def set_plot_legend(self, plot_legend=True, redraw_spectra=True):
        """
        Sets whether to plot legend for this spectrum or not.

        Parameters
        ----------
        plot_legend : bool
            Default True.
        redraw_spectra : bool
            If True (default), spectra will be redrawn.
        """
        self.plot_legend = plot_legend
        if redraw_spectra:
            self._redraw_all_spectra()

    def set_style(self, color=None, line_width=None, line_type=None, redraw_spectra=True):
        """
        Set color, line width and line type of plotted spectrum.

        Parameters
        ----------
        color : {str, tuple}, optional
            Color of the spectrum, use string for common colors (eg. 'black', 'blue', 'red', ...)
            or tuple - red, green, blue, alpha components from 0 - 255, eg. (255, 0, 0, 255).
            If None (default), user defined color scheme will be used.
        line_width : {int, float}, optional
            Sets the line width of the plotted line. If None (default), user defined color scheme will be used.
        line_type : int 0-6, optional
            Sets the line type of the plotted line. See https://doc.qt.io/archives/qt-4.8/qt.html#PenStyle-enum
            for line types. If None (default), user defined color scheme will be used.
        redraw_spectra : bool
            If True (default), spectra will be redrawn.
        """
        # if color is not None:
        self.color = color
        # if line_width is not None:
        self.line_width = line_width
        # if line_type is not None:
        self.line_type = line_type
        if redraw_spectra:
            # from user_namespace import redraw_all_spectra
            self._redraw_all_spectra()

    def set_default_style(self, redraw_spectra=False):
        """
        Sets `color`, `line_width` and `line_type` to None. User defined color scheme will be used.

        Parameters
        ----------
        redraw_spectra : bool
            If True (default), spectra will be redrawn.
        """
        self.color = None
        self.line_width = None
        self.line_type = None

        if redraw_spectra:
            # from user_namespace import redraw_all_spectra
            self._redraw_all_spectra()

    def x_values(self):
        """Returns the x values in this spectrum.

        :return: ndarray
        """
        return self.data[:, 0]

    def y_values(self):
        """Returns the y values in this spectrum.

        :return: ndarray
        """
        return self.data[:, 1]

    def length(self):
        """Returns the number of point that this spectrum has.

        :return: float
        """
        return self.data.shape[0]

    def x_min(self):
        """Returns the the first x value.

        :return: float
        """
        return self.data[0, 0]

    def x_max(self):
        """Returns the the last x value.

        :return: float
        """
        return self.data[-1, 0]

    def spacing(self):
        """
        Returns a type and spacing of the x values:
            * r - probably regular spacing for all points
            * i - probably irregular spacing for all points

        The r/i distinction is made by checking differences between first and the end x values.
        Value is computed as average spacing:  x_max - x_min / number of points

        :return: str
        """
        x_dif = self.data[-1, 0] - self.data[0, 0]
        spacing = x_dif / (self.data.shape[0] - 1)
        # regular spacing
        type = 'r'
        if self.data.shape[0] > 2:
            # probably irregular spacing, difference between first two and last two x values is not equal
            if not math.isclose(self.data[1, 0] - self.data[0, 0], self.data[-1, 0] - self.data[-2, 0]):
                type = 'i'

        return type + "{:.3g}".format(spacing)

    # https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
    def power_spectrum(self, redraw_spectra=True):
        """Calculates the power spectrum using FFT and replaces the data values. Power spectrum is normalized to 1.

        Parameters
        ----------
        redraw_spectra : bool
            If True (default), spectra will be redrawn (and Tree View updated).
        """

        n = self.data.shape[0]  # number of points in current spectrum

        vals = np.abs(fft(self.data[:, 1])) ** 2
        vals /= np.max(vals)
        freq = fftfreq(n, (self.data[-1, 0] - self.data[0, 0]) / (n - 1))

        filter = freq >= 0  # pick only frequencies >= 0

        output = np.vstack((freq[filter], vals[filter])).T

        self.data = output

        if redraw_spectra:
            self._redraw_all_spectra()
            self._update_view()

        return self

    def savgol(self, window_length, poly_order, redraw_spectra=True):
        """
        Applies Savitysky-Golay filter to a spectrum. Based on `scipy.signal.savgol_filter <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html>`_.

        Parameters
        ----------
        window_length : int
            Length of window that is used in the algorithm, must be positive odd integer.
        poly_order : int
            Polynomial order used in the algorithm, must be >= 1.
        redraw_spectra : bool
            If True (default), spectra will be redrawn (and Tree View updated).
        """
        window_length = int(window_length)
        poly_order = int(poly_order)

        if poly_order < 1:
            raise ValueError("Polynomial order must be > 0.")

        if poly_order >= window_length:
            raise ValueError("Polynomial order must be less than window_length.")

        # window_length must be odd number
        if window_length % 2 != 1:
            window_length += 1

        self.data[:, 1] = savgol_filter(self.data[:, 1], window_length, poly_order)

        if redraw_spectra:
            self._redraw_all_spectra()

        return self

    def differentiate(self, n=1, redraw_spectra=True):
        """
        Calculates a derivative spectrum and replaces the values. Length of the spectrum decreases by 1.

        Parameters
        ----------
        n : int
            The number of time s the values are differentiated. Default 1.
        redraw_spectra : bool
            If True (default), spectra will be redrawn (and Tree View updated).
        """

        new_data = np.zeros((self.data.shape[0] - n, 2))

        x_diffs = np.diff(self.data[:, 0], 1)[n - 1:]  # changes in x data

        new_data[:, 1] = np.diff(self.data[:, 1], n) / (x_diffs ** n)  # dy/dx
        new_data[:, 0] = self.data[n:, 0]

        self.data = new_data

        # self.data = Spectrum._differentiate(self.data)

        if redraw_spectra:
            self._redraw_all_spectra()
            self._update_view()
        return self

    #
    # @staticmethod
    # @jit(fastmath=True, nopython=True)
    # def _differentiate(data):
    #     der_data = np.zeros((data.shape[0] - 1, 2))
    #     der_data[:, 0] = data[1:, 0]
    #     for i in range(1, data.shape[0]):
    #         dx = data[i, 0] - data[i - 1, 0]
    #         der_data[i - 1, 1] = (data[i, 1] - data[i - 1, 1]) / dx  # dy/dx
    #
    #     return der_data

    def integrate(self, int_constant=0, redraw_spectra=True):
        """Integrates the spectrum using trapezoidal integration method, spectrum y values will
        be replaced by integrated values.

        Parameters
        ----------
        int_constant : {int, float}
            Integration constant, default 0.
        redraw_spectra : bool
            If True (default), spectra will be redrawn.
        """
        self.data[:, 1] = Spectrum._integrate_trap(self.data, int_constant)

        if redraw_spectra:
            self._redraw_all_spectra()

        return self

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _integrate_trap(data, int_constant):
        int_data = np.zeros(data.shape[0])
        int_data[0] = int_constant * 2
        sum = int_data[0]
        for i in range(1, data.shape[0]):
            dx = data[i, 0] - data[i - 1, 0]
            sum += (data[i, 1] + data[i - 1, 1]) * dx
            int_data[i] = sum

        int_data[:] /= 2

        return int_data

    def integral(self, x0=None, x1=None):
        """Calculate cumulative integral of spectrum at specified x range [x0, x1] by trapezoidal integration method.
        If x0, x1 are None, all spectrum will be integrated.

        Parameters
        ----------
        x0 : {int, float, None}
            First x value. If both x0 and x1 are None, all spectrum will be integrated.
        x1 : {int, float, None}
            Last x value. If both x0 and x1 are None, all spectrum will be integrated.

        Returns
        -------
        out : float
            Integral.
        """

        start_idx = 0
        end_idx = self.data.shape[0]  # not exactly end index, this is actually end_index + 1

        if x0 is not None or x1 is not None:
            if x0 >= x1:
                raise ValueError(f"Argument error, x0 ({x0}) cannot be larger or equal than x1 ({x1}).")

            start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
            end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        return np.trapz(self.data[start_idx:end_idx, 1], self.data[start_idx:end_idx, 0])

        # # cca 3 orders of magnitude faster than pythonic version
        # return Spectrum._integral_trap(self.data, start_idx, end_idx)

    #
    # @staticmethod
    # @jit(fastmath=True, nopython=True)
    # def _integral_trap(data, start, end):
    #     # perform trapezodic integration, assuming the data are unevenly distributed,
    #     # so no simplification in computation, except the division by 2 at the end
    #     # numba likes a lot of indexes
    #     sum = 0.0
    #     for i in range(start + 1, end):
    #         dx = data[i, 0] - data[i - 1, 0]
    #         sum += (data[i, 1] + data[i - 1, 1]) * dx
    #
    #     return sum / 2

    def baseline_correct(self, x0, x1, redraw_spectra=True):
        """Subtracts the average of y data from the specified x range [x0, x1] from y values.

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        redraw_spectra : bool
            If True (default), spectra will be redrawn.
        """

        if x0 > x1:
            raise ValueError(f"Argument error, x0 ({x0}) cannot be larger than x1 ({x1}).")

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        # calculate the average of y values over the selected range
        avrg = np.average(self.data[start_idx:end_idx, 1])

        # subtract the average from y values
        self.data[:, 1] -= avrg

        if redraw_spectra:
            self._redraw_all_spectra()

        return self

    def normalize(self, x0, x1, redraw_spectra=True):
        """Finds an y maximum at specified x range [x0, x1] and divide all y values by this maximum.

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        redraw_spectra : bool
            If True (default), spectra will be redrawn.
        """

        if x0 > x1:
            raise ValueError(f"Argument error, x0 ({x0}) cannot be larger than x1 ({x1}).")

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        y_max = np.max(self.data[start_idx:end_idx, 1])

        # normalize y values
        self.data[:, 1] /= y_max

        if redraw_spectra:
            self._redraw_all_spectra()

        return self

    def find_maximum(self, x0=None, x1=None):
        """Returns a point (x and y value), which belongs to a maximum y value in a specified x range [x0, x1].

        Parameters
        ----------
        x0 : {int, float, None}
            First x value. If both x0 and x1 are None, all spectrum will be searched.
        x1 : {int, float, None}
            Last x value. If both x0 and x1 are None, all spectrum will be searched.

        Returns
        -------
        out : tuple - (x, y)
            A highest point in spectrum.
        """
        start_idx = 0
        end_idx = self.data.shape[0]  # not exactly end index, this is actually end_index + 1

        if x0 is not None and x1 is not None:
            if x0 >= x1:
                raise ValueError(f"Argument error, x0 ({x0}) cannot be larger or equal than x1 ({x1}).")

            start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
            end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        max_idx = np.argmax(self.data[start_idx:end_idx, 1])

        # return the x a y max value
        return self.data[start_idx + max_idx, 0], self.data[start_idx + max_idx, 1]

    def interpolate(self, spacing=1, kind='linear', redraw_spectra=True):
        """Interpolates the spectrum, based on `scipy.interpolation.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_.

        Parameters
        ----------
        spacing : {int, float}
            Sets the spacing between output x values, default 1.
        kind : str
            Sets the kind of interpolation method used, eg. 'linear', 'quadratic', etc. For more, refer to
            `interp1d documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_.
            Default 'linear'.
        redraw_spectra : bool
            If True (default), spectra will be redrawn (and Tree View updated).
        """

        data_min = self.data[0, 0]
        data_max = self.data[-1, 0]

        if spacing > data_max - data_min:
            raise ValueError(f"Spacing ({spacing}) cannot be larger that data itself. "
                             f"Resample method, spectrum {self.name}.")

        # new x data must lie inside a range of original x values
        x_min = spacing * int(np.ceil(data_min / spacing))
        x_max = spacing * int(np.floor(data_max / spacing))

        # length of array of new data
        n = int((x_max - x_min) / spacing + 1)

        new_data = np.zeros((n, 2))
        new_data[:, 0] = np.linspace(x_min, x_max, num=n)  # new x values

        # interp1d returns a function that takes x values as the argument and returns the interpolated y values
        f = interp1d(self.data[:, 0], self.data[:, 1], kind=kind, copy=False, assume_sorted=True)

        new_data[:, 1] = f(new_data[:, 0])  # interpolate the y values

        # change the reference of self.data to new_data
        self.data = new_data

        if redraw_spectra:
            self._redraw_all_spectra()
            self._update_view()

        return self

    def cut(self, x0, x1, redraw_spectra=True):
        """Cuts the spectrum to x range [x0, x1].

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        redraw_spectra : bool
            If True (default), spectra will be redrawn (and Tree View updated).
        """
        if x0 >= x1:
            raise ValueError(f"Argument error, x0 ({x0}) cannot be larger or equal than x1 ({x1}).")

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        if start_idx + 1 == end_idx:
            raise ValueError(
                "Oh... someone likes to cut a lot, but with these input parameters, resulting spectrum would have"
                " 1 point :). Unfortunately, cannot perform cut operation.")

        self.data = self.data[start_idx:end_idx, :]

        if redraw_spectra:
            self._redraw_all_spectra()
            self._update_view()

        return self

    def extend_by_zeros(self, x0, x1, redraw_spectra=True):
        """
        Extends this spectrum to a new x range [x0, x1] by zeros. The original data will
        be replaced. Spacing will be determined from current data.

        Parameters
        ----------
        x0 : {int, float}
            New fist x value.
        x1 : {int, float}
            New last x value.
        redraw_spectra : bool
            If True (default), spectra will be redrawn.
        """
        if x0 >= x1:
            raise ValueError(f"Argument error, x0 ({x0}) cannot be larger or equal than x1 ({x1}).")

        x_min = self.x_min()
        x_max = self.x_max()

        # get nearby indexes of user defined x values
        start_idx = Spectrum.find_nearest_idx(self.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.data[:, 0], x1) + 1

        if start_idx != 0 and end_idx != self.data.shape[0]:
            # nothing to extend
            return
            # raise ValueError("Nothing to extend.")

        x_dif = x_max - x_min
        spacing = x_dif / (self.data.shape[0] - 1)

        new_x_min = spacing * int(np.round(x0 / spacing, 0))

        min_stack = None
        try:
            num_min = int((x_min - new_x_min) / spacing + 1)
            min_lin_space = np.linspace(new_x_min, x_min, num=num_min)

            min_stack = np.zeros((2, num_min - 1))
            min_stack[0] = min_lin_space[:-1]
            min_stack = min_stack.T
        except ValueError:
            pass

        max_stack = None
        try:
            new_x_max = spacing * int(np.round(x1 / spacing, 0))

            num_max = int((new_x_max - x_max) / spacing + 1)
            max_lin_space = np.linspace(x_max, new_x_max, num=num_max)

            max_stack = np.zeros((2, num_max - 1))
            max_stack[0] = max_lin_space[1:]
            max_stack = max_stack.T
        except ValueError:
            pass

        if min_stack is not None and max_stack is not None:
            result = np.vstack((min_stack, self.data, max_stack))
        elif min_stack is not None:
            result = np.vstack((min_stack, self.data))
        else:
            result = np.vstack((self.data, max_stack))

        self.data = result

        if redraw_spectra:
            self._redraw_all_spectra()
            self._update_view()

        return self

    def _operation_check(self, other):
        """Checks if the shapes of self and other data array are the same and where
        first and last x value is the same. If not, raises ValueError."""
        if not isinstance(other, Spectrum):
            return
        if other.data.shape[0] != self.data.shape[0]:
            raise ValueError(
                f"Spectra '{self.name}' and '{other.name}' have not the same length (dimension). "
                "Unable to perform calculation.")
        if other.data[0, 0] != self.data[0, 0] or other.data[-1, 0] != self.data[-1, 0]:
            raise ValueError(
                f"Spectra '{self.name}' and '{other.name}' have different x ranges. "
                "Unable to perform calculation.")

    def _arithmetic_operation(self, other, mode=CalcMode.add_sub):

        if not isinstance(other, Spectrum) and not (isinstance(other, float) or isinstance(other, int)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))

        shape = self.data.shape[0]

        # another spectrum
        if isinstance(other, Spectrum):
            self._operation_check(other)
            y_data = other.data[:, 1]

        else:  # number
            y_data = np.full(shape, other, dtype=np.float64)

        x_data = np.zeros(shape, dtype=np.float64) if mode == CalcMode.add_sub else np.full(shape, 1, dtype=np.float64)
        return np.vstack((x_data, y_data)).T

    def __add__(self, other):
        if isinstance(other, SpectrumList):
            return other.__radd__(self)
        other_data = self._arithmetic_operation(other, mode=CalcMode.add_sub)
        ret_data = self.data + other_data
        name = "{} + {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __sub__(self, other):
        if isinstance(other, SpectrumList):
            return other.__rsub__(self)
        other_data = self._arithmetic_operation(other, mode=CalcMode.add_sub)
        ret_data = self.data - other_data
        name = "{} - {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __mul__(self, other):
        if isinstance(other, SpectrumList):
            return other.__rmul__(self)
        other_data = self._arithmetic_operation(other, mode=CalcMode.mul_div)
        ret_data = self.data * other_data
        name = "{} * {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __truediv__(self, other):
        if isinstance(other, SpectrumList):
            return other.__rtruediv__(self)
        other_data = self._arithmetic_operation(other, mode=CalcMode.mul_div)
        ret_data = self.data / other_data
        name = "{} / {}".format(self.name, other.name if isinstance(other, Spectrum) else other)
        return Spectrum(ret_data, name=name)

    def __radd__(self, other):
        other_data = self._arithmetic_operation(other, mode=CalcMode.add_sub)
        ret_data = other_data + self.data
        name = "{} + {}".format(other.name if isinstance(other, Spectrum) else other, self.name)
        return Spectrum(ret_data, name=name)

    def __rsub__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))
        ret_data = self.data.copy()
        ret_data[:, 1] = other - ret_data[:, 1]
        name = "{} - {}".format(other, self.name)
        return Spectrum(ret_data, name=name)

    def __rmul__(self, other):
        other_data = self._arithmetic_operation(other, mode=CalcMode.mul_div)
        ret_data = other_data * self.data
        name = "{} * {}".format(other.name if isinstance(other, Spectrum) else other, self.name)
        return Spectrum(ret_data, name=name)

    def __rtruediv__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))
        ret_data = self.data.copy()
        ret_data[:, 1] = other / ret_data[:, 1]
        name = "{} / {}".format(other, self.name)
        return Spectrum(ret_data, name=name)

    def __neg__(self):
        sp = self * -1
        sp.name = f'-{self.name}'
        return sp

    def __pos__(self):
        return self.__copy__()

    def __pow__(self, power, modulo=None):  # self ** power
        if isinstance(power, SpectrumList):
            return power.__rpow__(self)

        if not isinstance(power, (Spectrum, float, int)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(power)))

        x_data = self.data[:, 0]

        # another spectrum
        if isinstance(power, Spectrum):
            self._operation_check(power)

            y_data = np.power(self.data[:, 1], power.data[:, 1])
            return Spectrum.from_xy_values(x_data, y_data, name=f'{self.name} ** {power.name}')

        else:  # number
            y_data = np.power(self.data[:, 1], power)
            return Spectrum.from_xy_values(x_data, y_data, name=f'{self.name} ** {power}')

    def __rpow__(self, other):  # other ** self
        if not isinstance(other, (Spectrum, float, int)):
            raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))

        x_data = self.data[:, 0]

        # another spectrum
        if isinstance(other, Spectrum):
            self._operation_check(other)

            y_data = np.power(other.data[:, 1], self.data[:, 1])
            return Spectrum.from_xy_values(x_data, y_data, name=f'{other.name} ** {self.name}')

        else:  # number
            y_data = np.power(other, self.data[:, 1])
            return Spectrum.from_xy_values(x_data, y_data, name=f'{other} ** {self.name}')

    @staticmethod
    def float_to_string(float_number, decimal_sep='.'):
        return str(float_number).replace('.', decimal_sep)

    def __str__(self, separator='\t', decimal_sep='.', new_line='\n', include_header=True):
        buffer = "Wavelength" + separator + self.name + new_line if include_header else ""
        buffer += new_line.join(
            separator.join(Spectrum.float_to_string(num, decimal_sep) for num in row) for row in self.data)
        return buffer

    @staticmethod
    def _list_to_stream(list_of_spectra, include_group_name=True, include_header=True, delimiter='\t',
                        decimal_sep='.', save_to_file=False, dir_path=None, extension=None,
                        x_data_name='Wavelength / nm'):

        # this generator iterates the ndarray data and yield returns a list of formatted row for csv writer
        def iterate_data(iterable):
            for row in iterable:
                yield [str(num).replace('.', decimal_sep) for num in row]

        if not isinstance(list_of_spectra, list):
            raise ValueError("Argument 'list_of_spectra' must be type of list.")

        buffer = ""

        dialect = csv.excel
        dialect.delimiter = delimiter
        dialect.lineterminator = '\n'
        dialect.quoting = csv.QUOTE_MINIMAL

        for i, node in enumerate(list_of_spectra):

            if save_to_file:
                name = node.name if isinstance(node, Spectrum) else node[0].group_name
                filepath = os.path.join(dir_path,
                                        (name if name != '' else 'Untitled{}'.format(i)) + extension)
                w = csv.writer(open(filepath, 'w', encoding='utf-8'), dialect=dialect)
            else:
                stream = io.StringIO('')
                w = csv.writer(stream, dialect=dialect)

            # export as group
            if isinstance(node, list):
                if len(node) == 0:
                    continue

                if include_group_name:
                    w.writerow([node[0].group_name])

                # add header if it is user defined
                if include_header:
                    w.writerow([x_data_name] + [sp.name for sp in node])

                # add row of wavelength, then, we will transpose the matrix, otherwise, we would have to reshape
                matrix = node[0].data[:, 0]

                # add absorbance data to matrix from all exported spectra
                for sp in node:
                    if sp.length() != node[0].length():
                        raise ValueError(
                            f"Spectra '{node[0].name}' and '{sp.name}' in group '{node[0].group_name}'"
                            f" have not the same length (dimension). Unable to export.")
                    if sp.data[0, 0] != node[0].data[0, 0] or sp.data[-1, 0] != node[0].data[-1, 0]:
                        raise ValueError(
                            f"Spectra '{node[0].name}' and '{sp.name}' in group '{node[0].group_name}'"
                            f" have not the same length (dimension). Unable to export.")
                    matrix = np.vstack((matrix, sp.data[:, 1]))

                matrix = matrix.T  # transpose

                # write matrix
                w.writerows(iterate_data(matrix))

            # export as single spectrum file
            if isinstance(node, Spectrum):
                if include_header:
                    w.writerow([x_data_name, node.name])

                w.writerows(iterate_data(node.data))

            if not save_to_file:
                buffer += stream.getvalue() + '\n'
                stream.close()

        if buffer != "":
            # return ret_buffer and remove the 2 new line characters that are at the end
            return buffer[:-2]

    @staticmethod
    def list_to_files(list_of_spectra, dir_path, extension, include_group_name=True, include_header=True,
                      delimiter='\t',
                      decimal_sep='.', x_data_name='Wavelength / nm'):

        Spectrum._list_to_stream(list_of_spectra, include_group_name, include_header, delimiter,
                                 decimal_sep, True, dir_path, extension, x_data_name)

    @staticmethod
    def list_to_string(list_of_spectra, include_group_name=True, include_header=True, delimiter='\t',
                       decimal_sep='.', x_data_name='Wavelength / nm'):

        return Spectrum._list_to_stream(list_of_spectra, include_group_name, include_header, delimiter,
                                        decimal_sep, False, x_data_name=x_data_name)

    @staticmethod
    def find_nearest_idx(array, value):
        """
        Finds nearest index of `value` in `array`. If value >  max(array), the last index of array
        is returned, if value < min(array), 0 is returned. Array must be sorted. Also works for value
        to be array, then array of indexes is returned.

        Parameters
        ----------
        array : ndarray
            Array to be searched.
        value : {int, float}
            Value.

        Returns
        -------
        out : int, np.ndarray
            Found nearest index/es to value/s.
        """
        # idx = np.searchsorted(array, value, side="left")
        # if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        #     return idx - 1
        # else:
        #     return idx
        if isinstance(value, (int, float)):
            value = np.asarray([value])
        else:
            value = np.asarray(value)

        result = np.empty_like(value, dtype=int)
        for i in range(value.shape[0]):
            idx = np.searchsorted(array, value[i], side="left")
            if idx > 0 and (
                    idx == len(array) or math.fabs(value[i] - array[idx - 1]) < math.fabs(value[i] - array[idx])):
                result[i] = idx - 1
            else:
                result[i] = idx
        return result if result.shape[0] > 1 else result[0]

    @staticmethod
    def find_nearest(array, value):
        """
        Returns closest value in in `array`. If value >  max(array), the last value of array
        is returned, if value < min(array), first value of array is returned. Array must be sorted.

        Parameters
        ----------
        array : ndarray
            Array to be searched.
        value : {int, float}
            Value.

        Returns
        -------
        out : type of values in `array`
            Found nearest value.
        """
        idx = Spectrum.find_nearest_idx(array, value)
        return array[idx]

    def __copy__(self):
        """Deep copy the current instance as Spectrum object."""
        sp = Spectrum(None, filepath=self.filepath, name=self.name, group_name=self.group_name, color=self.color,
                      line_width=self.line_width, line_alpha=self.line_alpha, line_type=self.line_type, symbol=self.symbol,
                      symbol_brush=self.symbol_brush, sym_brush_alpha=self.sym_brush_alpha, symbol_fill=self.symbol_fill,
                      sym_fill_alpha=self.sym_fill_alpha, symbol_size=self.symbol_size, plot_legend=self.plot_legend)

        sp.data = deepcopy(self.data)
        return sp


class SpectrumList(object):

    def __init__(self, children=None, name=''):
        """
        Class that holds a group of spectra as a list and enables simple arithmetic calculations among groups of spectra, number and
        group and spectrum and group. It acts as a list, so the individual spectra can be accessed by putting square brackets
        after the instance of an object and putting appropriate index in them. Also, spectrum objects can be iterated the same way as list.
        Also, it enables other calculations on the group of spectra, like baseline correction, cutting, basically all the
        operation provided in :class:`Spectrum` class plus additional operations that can
        be only used for groups, like transposition, etc. ....

        Parameters
        ----------
        children : {list, None}
            List of :class:`Spectrum`, default None.
        name : str, optional
            The group name.

        Attributes
        ----------
        children : list of :class:`Spectrum`
            List of members in this group.
        name : str
            The group name as appeared in Tree Widget.
        """

        self.children = [] if children is None else children
        self.name = name

    def __len__(self):
        return self.children.__len__()

    def add_to_list(self, spectra=None):
        """Adds this group to the Tree Widget. Only works in SSM's Console Widget.

        Parameters
        ----------
        spectra : {list, None}
            If None, current members will be added, otherwise, spectra will be imported.
        """
        try:
            from user_namespace import add_to_list
            add_to_list(self if spectra is None else spectra)
        except ImportError:
            pass

    def _redraw_all_spectra(self):
        pass

    def _update_view(self):
        pass

    def get_names(self):
        """Returns names of all Spectrum objects as a list.

        :returns: list
        """

        return [sp.name for sp in self._iterate_items()]

    def set_names(self, names):
        """Sets the names of the group and updates the Tree Widget and redraws the spectra.

        Parameters
        ----------
        names : iterable, eg. list, tuple
            New names that will be replaced. The length of names can be different from number
            of spectra in group (using a zip function).
        """

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        for sp, new_name in zip(self, names):
            sp.name = str(new_name)

        # from user_namespace import update_view, redraw_all_spectra
        self._update_view()
        self._redraw_all_spectra()

    def set_plot_legend(self, plot_legend=True):
        """
        Sets whether to plot legend for this group or not and redraws all spectra.

        Parameters
        ----------
        plot_legend : bool
            Default True.
        """

        for sp in self._iterate_items():
            sp.set_plot_legend(plot_legend, False)

        self._redraw_all_spectra()

    def set_style(self, color=None, line_width=None, line_type=None, redraw_spectra=True):
        """
        Sets color, line width and line type of all group and redraws all spectra.

        Parameters
        ----------
        color : {str, tuple}, optional
            Color of the spectrum, use string for common colors (eg. 'black', 'blue', 'red', ...)
            or tuple - red, green, blue, alpha components from 0 - 255, eg. (255, 0, 0, 255).
            If None (default), user defined color scheme will be used.
        line_width : {int, float}, optional
            Sets the line width of the plotted line. If None (default), user defined color scheme will be used.
        line_type : int 0-6, optional
            Sets the line type of the plotted line. See https://doc.qt.io/archives/qt-4.8/qt.html#PenStyle-enum
            for line types. If None (default), user defined color scheme will be used.
        """

        for sp in self._iterate_items():
            sp.set_style(color, line_width, line_type, False)
        if redraw_spectra:
            self._redraw_all_spectra()

    def _iterate_items(self):
        for node in self:
            if isinstance(node, SpectrumList):
                for sp in node:
                    if not isinstance(sp, Spectrum):
                        raise ValueError("Objects in list have to be type of Spectrum.")
                    yield sp
                continue
            if isinstance(node, Spectrum):
                yield node
                continue
            raise ValueError("Objects in list have to be type of Spectrum.")

    def differentiate(self, n=1):
        """
        Calculates a derivatives for this group, replaces the values and redraws the spectra.
        Length of the spectrum decreases by 1.
        """
        for sp in self._iterate_items():
            sp.differentiate(n, False)

        self._redraw_all_spectra()
        self._update_view()

        return self

    def integrate(self, int_constant=0):
        """Integrates the group of spectra using trapezoidal integration method, spectra's y values will
        be replaced by integrated values and redraws the spectra.

        Parameters
        ----------
        int_constant : {int, float}
            Integration constant, default 0.
        """

        for sp in self._iterate_items():
            sp.integrate(int_constant, False)

        self._redraw_all_spectra()

        return self

    def integral(self, x0=None, x1=None):
        """Calculates cumulative integral for each spectrum in this group at specified x range [x0, x1] by trapezoidal integration method.
        If x0, x1 are None, all spectra will be integrated. Result is returned as ndarray.

        Parameters
        ----------
        x0 : {int, float, None}
            First x value. If both x0 and x1 are None, all spectra will be integrated.
        x1 : {int, float, None}
            Last x value. If both x0 and x1 are None, all spectra will be integrated.

        Returns
        -------
        out : numpy.ndarray
            Array of integrals.
        """

        results = []

        for sp in self._iterate_items():
            results.append(sp.integral(x0, x1))

        return np.asarray(results, dtype=np.float64)

    def find_maximum(self, x0=None, x1=None):
        """Returns an array of points (x and y value), which belongs to a maximum y value in a specified x range [x0, x1].
        First column contains x values, the second y values.

        Parameters
        ----------
        x0 : {int, float, None}
            First x value. If both x0 and x1 are None, all spectra will be searched.
        x1 : {int, float, None}
            Last x value. If both x0 and x1 are None, all spectra will be searched.

        Returns
        -------
        out : ndarray
            A highest points in spectra.
        """
        results = []

        for sp in self._iterate_items():
            results.append(sp.find_maximum(x0, x1))

        return np.asarray(results, dtype=np.float64)

    def get_y_values_at_x(self, x):
        """Returns y values at particular x value as an ndarray in this group of spectra.

        Parameters
        ----------
        x : {int, float}
            The x value.

        Returns
        -------
        out : ndarray
            Array of y values at this x value.
        """

        ret_list = []

        for sp in self._iterate_items():
            idx = Spectrum.find_nearest_idx(sp.data[:, 0], x)
            ret_list.append(sp.data[idx, 1])

        return np.asarray(ret_list, dtype=np.float64)

    def transpose(self, max_items=1000):
        """
        Transposes group, names of spectra in group will be taken as x values for transposed data, so these values
        must be convertible to int or float numbers. No text is allowed in these cells, only values.
        All x values of spectra in the group will become names in new group. The transposed group is added to Tree Widget.
        The operation is basically the same as copying the group of spectra to Excel, performs transposition of the matrix
        and copying back to SSM.

        Parameters
        ----------
        max_items : int
            Maximum number of items that will be added. If the transposition produce more than *max_items*,
            ValueError will be raised. Default 1000.
        """

        if len(self) < 2:
            raise ValueError("At least 2 items have to be in the group in order to perform transposition.")

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        x_vals_temp = []

        for sp in self:
            try:
                x_vals_temp.append(float(sp.name.replace(',', '.').strip()))
            except ValueError:
                raise ValueError("Names of spectra cannot be parsed to float.")

        x_values = np.asarray(x_vals_temp, dtype=np.float64)
        matrix = self[0].data[:, 0]  # wavelengths

        # add absorbance data to matrix from all exported spectra
        for sp in self:
            if sp.length() != self[0].length():
                raise ValueError(
                    "Spectra \'{}\' and \'{}\' have not the same length (dimension). "
                    "Unable to transpose.".format(self[0].name, sp.name))
            matrix = np.vstack((matrix, sp.data[:, 1]))

        group_name = "Transpose of {}".format(self.name)
        spectra = []

        if matrix.shape[1] > max_items:
            raise ValueError(
                "Number of transposed items ({}) exceeded the maximum number ({}). Cannot transpose spectra.".format(
                    matrix.shape[1], max_items))

        for i in range(matrix.shape[1]):
            sp = Spectrum.from_xy_values(x_values, matrix[1:, i], name=str(matrix[0, i]), group_name=group_name)
            spectra.append(sp)

        self.add_to_list([spectra])

    def power_spectrum(self):
        """Calculates the power spectrum using FFT and replaces the data values and redraws the spectra. Power spectrum is normalized to 1.
        """

        for sp in self._iterate_items():
            sp.power_spectrum(False)

        self._redraw_all_spectra()
        self._update_view()

        return self

    def savgol(self, window_length, poly_order):
        """
        Applies Savitysky-Golay filter to a group of spectra and redraws the spectra. Based on `scipy.signal.savgol_filter <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html>`_.

        Parameters
        ----------
        window_length : int
            Length of window that is used in the algorithm, must be positive odd integer.
        poly_order : int
            Polynomial order used in the algorithm, must be >= 1.
        """

        for sp in self._iterate_items():
            sp.savgol(window_length, poly_order, False)

        self._redraw_all_spectra()

        return self

    def baseline_correct(self, x0, x1):
        """Subtracts the average of y data from the specified x range [x0, x1] from y values for each spectrum in
        this group and redraws the spectra.

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """
        for sp in self._iterate_items():
            sp.baseline_correct(x0, x1, False)

        self._redraw_all_spectra()

        return self

    def cut(self, x0, x1):
        """Cuts all spectra in this group to x range [x0, x1] and redraws the spectra.

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """
        for sp in self._iterate_items():
            sp.cut(x0, x1, False)

        self._redraw_all_spectra()
        self._update_view()

        return self

    def interpolate(self, spacing=1, kind='linear'):
        """Interpolates the spectra in this group and redraws them, based on `scipy.interpolation.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_.

        Parameters
        ----------
        spacing : {int, float}
            Sets the spacing between output x values, default 1.
        kind : str
            Sets the kind of interpolation method used, eg. 'linear', 'quadratic', etc. For more, refer to
            `interp1d documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_.
            Default 'linear'.
        """
        for sp in self._iterate_items():
            sp.interpolate(spacing, kind, False)

        self._redraw_all_spectra()
        self._update_view()

        return self

    def normalize(self, x0, x1):
        """Finds an y maximum at specified x range [x0, x1] and divide all y values by this maximum. This
        operation is perfomred for each spectrum in this group. The spectra are redrawn.

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """
        for sp in self._iterate_items():
            sp.normalize(x0, x1, False)

        self._redraw_all_spectra()

        return self

    def extend_by_zeros(self, x0, x1):
        """
        Extends spectra in this group to a new x range [x0, x1] by zeros and redraws the spectra. The original data will
        be replaced. Spacing will be determined from data of current spectrum.

        Parameters
        ----------
        x0 : {int, float}
            New fist x value.
        x1 : {int, float}
            New last x value.
        """
        for sp in self._iterate_items():
            sp.extend_by_zeros(x0, x1, False)

        self._redraw_all_spectra()
        self._update_view()

        return self

    def _arithmetic_operation(self, other, func_operation):
        """
        Perform an defined operation on groups of spectra (type Spectrum). func_operation is a pointer to function.
        This function takes 2 arguments that must be type of Spectrum.
        """
        # operation with another list, group + group
        if isinstance(other, SpectrumList):
            if len(self) != len(other):
                raise ValueError("Cannot perform an operation on groups which contains different number of items.")
            if len(self) == 0:
                return SpectrumList()
            if not isinstance(self[0], Spectrum) or not isinstance(other[0], Spectrum):
                raise ValueError("Objects in list have to be type of Spectrum.")
            ret_list = SpectrumList()
            for i in range(len(self)):
                ret_list.children.append(func_operation(self[i], other[i]))
            return ret_list, other.name

        # operation with single spectrum, group + spectrum or with number, eg. group - 1
        if isinstance(other, (Spectrum, float, int)):
            if len(self) == 0:
                return SpectrumList()

            if not isinstance(self[0], Spectrum):
                raise ValueError("Objects in list have to be type of Spectrum.")
            ret_list = SpectrumList()
            for sp in self:
                ret_list.children.append(func_operation(sp, other))
            return ret_list, str(other) if isinstance(other, (float, int)) else other.name

        raise ValueError("Cannot perform calculation of SpectrumList with {}".format(type(other)))

    def __add__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 + s2)
        list.name = "{} + {}".format(self.name, other_str)
        return list

    def __sub__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 - s2)
        list.name = "{} - {}".format(self.name, other_str)
        return list

    def __mul__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 * s2)
        list.name = "{} * {}".format(self.name, other_str)
        return list

    def __truediv__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 / s2)
        list.name = "{} / {}".format(self.name, other_str)
        return list

    def __radd__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 + s1)
        list.name = "{} + {}".format(other_str, self.name)
        return list

    def __rsub__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 - s1)
        list.name = "{} - {}".format(other_str, self.name)
        return list

    def __rmul__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 * s1)
        list.name = "{} * {}".format(other_str, self.name)
        return list

    def __rtruediv__(self, other):
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 / s1)
        list.name = "{} / {}".format(other_str, self.name)
        return list

    def __neg__(self):
        list = SpectrumList()
        list.name = f'-{self.name}'
        for sp in self:
            list.children.append(-sp)
        return list

    def __pos__(self):
        return self.__copy__()

    def __pow__(self, power, modulo=None):  # self ** power
        list, other_str = self._arithmetic_operation(power, lambda s1, s2: s1 ** s2)
        list.name = "{} ** {}".format(self.name, other_str)
        return list

    def __rpow__(self, other):  # other ** self
        list, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 ** s1)
        list.name = "{} ** {}".format(other_str, self.name)
        return list

    def __getitem__(self, item):
        return self.children[item]

    def __iter__(self):
        return iter(self.children)

    def __copy__(self):
        """Deep copy this instance as SpectrumList."""
        ret = SpectrumList(name=self.name)
        for child in self.children:
            ret.children.append(child.__copy__())
        return ret
