
import numpy as np
import math

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq
from scipy.integrate import cumtrapz, simps

# import functools


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
    idx = find_nearest_idx(array, value)
    return array[idx]


def group2mat(spectra):
    """Given a generator of spectra, it returns the x, 'name' values and matrix.
    The columns of data will remain columns are horizontaly stack together.

    Parameters
    ----------
    spectra : iterable
        Spectra.

    Returns
    -------
    out : tuple
        x_values, converted_names as y values, matrix with [x, y] dimension"""

    header_vals_temp = []

    x = None
    matrix = None
    err = False

    for sp in spectra:
        try:
            header_vals_temp.append(float(sp.name.replace(',', '.').strip()))
        except ValueError:
            err = True

        if x is None:
            x = sp.data[:, 0]
        else:
            if x.shape[0] != sp.data.shape[0]:
                raise ValueError("Spectra do not have the same dimension within the group. Unable to perform the operation.")
            if not np.allclose(x, sp.data[:, 0]):
                raise ValueError("Spectra do not have the same x values within the group. Unable to perform the operation.")

        matrix = sp.data[:, 1] if matrix is None else np.vstack((matrix, sp.data[:, 1]))

    y = None if err else np.asarray(header_vals_temp)

    return x, y, matrix.T


class Spectrum(object):

    modif_funcs = ['power_spectrum',
                   'savgol',
                   'gradient',
                   'differentiate',
                   'integrate',
                   'interpolate',
                   'baseline_correct',
                   'normalize',
                   'cut',
                   'extend_by_zeros']

    op_funcs = ['find_maximum', 'integral']

    def __init__(self, data, name='', filepath=None, assume_sorted=False):
        """
        Class that holds the spectrum object as a 2D array (dimensions n x 2) where n is a number
        of points in spectrum and includes various functions used for data manipulation with them.
        Spectrum is stored in variable *data* as a numpy ndarray.

        Parameters
        ----------
        data : numpy.ndarray
            A raw data of spectrum object, dimensions are (n x 2) where n is a number of points in spectrum.
            First column represents *x* values, second column *y* values.
        name : str
            The name of the spectrum as is displayed in Tree Widget.
        filepath : {str, None}
            If the spectrum was imported from file, this variable stored the path to the file it was imported from.
        """

        if assume_sorted:
            self.data = np.asarray(data, dtype=np.float64) if data is not None else None
        else:
            # sort according to first column,numpy matrix, 1. column wavelength, 2. column absorbance
            self.data = np.asarray(data[data[:, 0].argsort()], dtype=np.float64) if data is not None else None

        self.filepath = filepath
        self.name = name

    @classmethod
    def from_xy_values(cls, x_values, y_values, name=''):
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

        return cls(data, name=name)

    @property
    def x(self):
        """Returns the x values of this spectrum.

        :return: ndarray
        """
        return self.data[:, 0]

    @x.setter
    def x(self, array):
        """Sets the x values of this spectrum.

        :return: ndarray
        """
        self.data[:, 0] = array

    @property
    def y(self):
        """Returns the y values of this spectrum.

        :return: ndarray
        """
        return self.data[:, 1]

    @y.setter
    def y(self, array):
        """Sets the y values of this spectrum.

        :return: ndarray
        """
        self.data[:, 1] = array

    def length(self):
        """Returns the number of point that this spectrum has.

        :return: float
        """
        return self.data.shape[0]

    def power_spectrum(self):
        # https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
        """Calculates the power spectrum using FFT and replaces the data values. Power spectrum is normalized to 1.
        """

        n = self.data.shape[0]  # number of points in current spectrum

        vals = np.abs(fft(self.data[:, 1])) ** 2
        vals /= np.max(vals)
        freq = fftfreq(n, (self.data[-1, 0] - self.data[0, 0]) / (n - 1))

        filter = freq >= 0  # pick only frequencies >= 0

        output = np.vstack((freq[filter], vals[filter])).T

        self.data = output

        return self

    def savgol(self, window_length, poly_order):
        """
        Applies Savitsky-Golay filter to a spectrum. Based on `scipy.signal.savgol_filter <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html>`_.

        Parameters
        ----------
        window_length : int
            Length of window that is used in the algorithm, must be positive odd integer.
        poly_order : int
            Polynomial order used in the algorithm, must be >= 1.
        """
        window_length = int(window_length)
        poly_order = int(poly_order)

        if poly_order < 1:
            raise ValueError("Polynomial order must be > 0.")

        if poly_order >= window_length:
            raise ValueError("Polynomial order must be less than window_length.")

        # window_length must be odd number
        window_length += int(not window_length % 2)

        self.data[:, 1] = savgol_filter(self.data[:, 1], window_length, poly_order)

        return self

    def gradient(self, edge_order=1):
        """
        Calculates a gradient using central differences and replaces the original values.

        Parameters
        ----------
        edge_order : int
            1 or 2, Gradient is calculated using N-th order accurate differences at the boundaries. Default: 1.
        """

        self.data[:, 1] = np.gradient(self.data[:, 1], self.data[:, 0], edge_order=edge_order)

        return self

    def differentiate(self, n=1):
        """
        Calculates a derivative spectrum and replaces the values. Length of the spectrum decreases by 1.

        Parameters
        ----------
        n : int
            The number of time s the values are differentiated. Default 1.
        """

        new_data = np.zeros((self.data.shape[0] - n, 2))

        x_diffs = np.diff(self.data[:, 0], 1)[n - 1:]  # changes in x data

        new_data[:, 1] = np.diff(self.data[:, 1], n) / (x_diffs ** n)  # dy/dx
        new_data[:, 0] = self.data[n:, 0]

        self.data = new_data

        return self

    def integrate(self, int_constant=0):
        """Integrates the spectrum using treapezoidal integration method, spectrum y values will
        be replaced by integrated values.

        Parameters
        ----------
        int_constant : {int, float}
            Integration constant, default 0.
        """
        self.data[:, 1] = cumtrapz(self.data[:, 1], self.data[:, 0], initial=int_constant)

        return self

    def _get_xy_from_range(self, x0=None, x1=None):
        start_idx = 0
        end_idx = self.data.shape[0]

        if x0 is not None and x1 is not None and x0 > x1:
            x0, x1 = x1, x0

        if x0 is not None:
            start_idx = find_nearest_idx(self.data[:, 0], x0)

        if x1 is not None:
            end_idx = find_nearest_idx(self.data[:, 0], x1) + 1

        x = self.data[start_idx:end_idx, 0]
        y = self.data[start_idx:end_idx, 1]

        return x, y

    def integral(self, x0=None, x1=None, method: str = 'trapz'):
        """Calculate cumulative integral of spectrum at specified x range [x0, x1] by trapezoidal integration method.
        If x0, x1 are None, all spectrum will be integrated.

        Parameters
        ----------
        x0 : {int, float, None}
            First x value. If both x0 and x1 are None, all spectrum will be integrated.
        x1 : {int, float, None}
            Last x value. If both x0 and x1 are None, all spectrum will be integrated.

        method : str
            this can be sum, trapz or simps
            default trapz

        Returns
        -------
        out : float
            Integral - scalar.
        """

        x, y = self._get_xy_from_range(x0, x1)

        if method == 'sum':
            x_diff = x[1:] - x[:-1]
            return y[:-1].sum() * x_diff + y[-1]
        elif method == 'trapz':
            return np.trapz(y, x)
        else:  # simpsons rule
            return simps(y, x)

    def baseline_correct(self, x0=None, x1=None):
        """Subtracts the average of y data from the specified x range [x0, x1] from y values.

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """

        x, y = self._get_xy_from_range(x0, x1)

        # subtract the average from y values
        self.data[:, 1] -= np.average(y)

        return self

    def normalize(self, x0=None, x1=None):
        """Finds an y maximum at specified x range [x0, x1] and divide all y values by this maximum.

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """

        x, y = self._get_xy_from_range(x0, x1)

        # normalize y values
        self.data[:, 1] /= y.max()

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

        x, y = self._get_xy_from_range(x0, x1)

        max_idx = np.argmax(y)

        # return the x a y max value
        return x[max_idx], y[max_idx]

    def interpolate(self, spacing=1, kind='linear'):
        """Interpolates the spectrum, based on
        `scipy.interpolation.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_.

        Parameters
        ----------
        spacing : {int, float}
            Sets the spacing between output x values, default 1.
        kind : str
            Sets the kind of interpolation method used, eg. 'linear', 'quadratic', etc. For more, refer to
            `interp1d documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_.
            Default 'linear'.
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

        return self

    def cut(self, x0=None, x1=None):
        """Cuts the spectrum to x range [x0, x1].

        Parameters
        ----------
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """

        if x0 is None and x1 is None:  # nothing to cut
            return self

        x, y = self._get_xy_from_range(x0, x1)

        self.data = np.vstack((x, y)).T

        return self

    def extend_by_zeros(self, x0=None, x1=None):
        """
        Extends this spectrum to a new x range [x0, x1] by zeros. The original data will
        be replaced. Spacing will be determined from current data.

        Parameters
        ----------
        x0 : {int, float}
            New fist x value.
        x1 : {int, float}
            New last x value.
        """

        if x0 is None and x1 is None:
            return self

        start_idx = 0
        end_idx = self.data.shape[0]

        if x0 is not None and x1 is not None and x0 > x1:
            x0, x1 = x1, x0

        if x0 is not None:
            start_idx = find_nearest_idx(self.data[:, 0], x0)

        if x1 is not None:
            end_idx = find_nearest_idx(self.data[:, 0], x1) + 1

        if start_idx != 0 and end_idx != self.data.shape[0]:
            # nothing to extend
            return self

        x_min = self.data[0, 0]
        x_max = self.data[-1, 0]

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

        return self

    def _operation_check(self, other):
        """Checks if the shapes of self and other data array are the same and where
        first and last x value is the same. If not, raises ValueError."""
        if self.data.shape[0] != other.data.shape[0] or not np.allclose(self.data[:, 0], other.data[:, 0]):
            raise ValueError(
                f"Spectra '{self.name}' and '{other.name}' does not share the same x values "
                f"(or have different dimension). Unable to perform operation.")

    def _arithmetic_operation(self, other, func_operation):
        # another spectrum
        if isinstance(other, Spectrum):
            self._operation_check(other)
            y_data = func_operation(self.data[:, 1], other.data[:, 1])
            other_str = other.name

        # list, tuple or ndarray that has the same dimension as current spectrum
        elif isinstance(other, (list, tuple, np.ndarray)):
            y = np.asarray(other)
            if y.shape[0] != self.data.shape[0]:
                raise ValueError(
                    f"Spectrum '{self.name}' and data does not have the same dimension. Unable to perform operation.")

            y_data = func_operation(self.data[:, 1], y)
            other_str = str(type(other))

        else:  # number
            y_data = func_operation(self.data[:, 1], other)
            other_str = str(other)

        # else:
        #     raise ValueError("Cannot perform calculation of Spectrum with {}".format(type(other)))

        return Spectrum.from_xy_values(self.data[:, 0], np.nan_to_num(y_data)), other_str

    def __add__(self, other):
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 + s2)
        sp.name = f"{self.name} + {other_str}"

        return sp

    def __sub__(self, other):
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 - s2)
        sp.name = f"{self.name} - {other_str}"

        return sp

    def __mul__(self, other):
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 * s2)
        sp.name = f"{self.name} * {other_str}"

        return sp

    def __truediv__(self, other):   # self / other
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s1 / s2)
        sp.name = f"{self.name} / {other_str}"

        return sp

    def __radd__(self, other):   # other + self
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 + s1)
        sp.name = f"{other_str} + {self.name}"

        return sp

    def __rsub__(self, other):   # other - self
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 - s1)
        sp.name = f"{other_str} - {self.name}"

        return sp

    def __rmul__(self, other):   # other * self
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 * s1)
        sp.name = f"{other_str} * {self.name}"

        return sp

    def __rtruediv__(self, other):  # other / self
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 / s1)
        sp.name = f"{other_str} / {self.name}"

        return sp

    def __pow__(self, power, modulo=None):  # self ** power
        if not isinstance(power, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented
        sp, other_str = self._arithmetic_operation(power, lambda s1, s2: s1 ** s2)
        sp.name = f"{self.name} ** {other_str}"

        return sp

    def __rpow__(self, other):  # other ** self
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return NotImplemented

        sp, other_str = self._arithmetic_operation(other, lambda s1, s2: s2 ** s1)
        sp.name = f"{other_str} ** {self.name}"

        return sp

    def __neg__(self):
        sp = self * -1
        sp.name = f'-{self.name}'
        return sp

    def __pos__(self):
        return self.__copy__()

    def __str__(self, separator='\t', decimal_sep='.', new_line='\n', include_header=True):
        if self.data is None:
            return 'Spectrum without data'
        return f"Spectrum {self.name}, x_range=({self.data[0, 0]}, {self.data[-1, 0]}), n_points={self.data.shape[0]}"
        # buffer = "Wavelength" + separator + self.name + new_line if include_header else ""
        # buffer += new_line.join(
        #     separator.join(str(num).replace('.', decimal_sep) for num in row) for row in self.data)
        # return buffer

    def __copy__(self):
        """Deep copy the current instance as Spectrum object."""
        sp = Spectrum(None, filepath=self.filepath, name=self.name)

        sp.data = self.data.copy()
        return sp


class SpectrumList:

    # def __new__(cls):
    #     self = super(SpectrumList, cls).__new__(cls)

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

        # setup functions that modify spectra
        for mf in Spectrum.modif_funcs:

            # # exec(f"def _func(*args):\n\tfor sp in self:\n\t\tsp.{mf}(*args)\n\n\treturn self", locals(), globals())
            # exec(f"def func(): return 1")

            mf_func = getattr(Spectrum, mf)  # get static function declaration

            def _func(*args, **kwargs):
                for sp in self:
                    mf_func(sp, *args, **kwargs)

                return self

            _func.__doc__ = mf_func.__doc__
            setattr(self, mf, _func)

        # setup functions that returns some result
        for of in Spectrum.op_funcs:
            of_func = getattr(Spectrum, of)  # get static function declaration

            def _func(*args, **kwargs):
                results = []
                for sp in self:
                    results.append(of_func(sp, *args, **kwargs))

                return np.asarray(results, dtype=np.float64)

            _func.__doc__ = of_func.__doc__
            setattr(self, of, _func)

        # exec("def afunc(): return 1")
        # setattr(self, 'mf', afunc)

    def __len__(self):
        return self.children.__len__()

    def get_names(self):
        """Returns names of all Spectrum objects as a list.

        :returns: list
        """

        return [sp.name for sp in self]

    # update all spectra, view
    def set_names(self, names):
        """Sets the names of the group and updates the Tree Widget and redraws the spectra.

        Parameters
        ----------
        names : iterable, eg. list, tuple
            New names that will be replaced. The length of names can be different from number
            of spectra in group (using a zip function).
        """

        for sp, new_name in zip(self, names):
            sp.name = str(new_name)

    # def set_plot_legend(self, plot_legend=True):
    #     """
    #     Sets whether to plot legend for this group or not and redraws all spectra.
    #
    #     Parameters
    #     ----------
    #     plot_legend : bool
    #         Default True.
    #     """
    #
    #     for sp in self:
    #         sp.set_plot_legend(plot_legend, False)
    #
    # #     self._redraw_all_spectra()
    #
    # def set_style(self, color=None, line_width=None, line_type=None, redraw_spectra=True):
    #     """
    #     Sets color, line width and line type of all group and redraws all spectra.
    #
    #     Parameters
    #     ----------
    #     color : {str, tuple}, optional
    #         Color of the spectrum, use string for common colors (eg. 'black', 'blue', 'red', ...)
    #         or tuple - red, green, blue, alpha components from 0 - 255, eg. (255, 0, 0, 255).
    #         If None (default), user defined color scheme will be used.
    #     line_width : {int, float}, optional
    #         Sets the line width of the plotted line. If None (default), user defined color scheme will be used.
    #     line_type : int 0-6, optional
    #         Sets the line type of the plotted line. See https://doc.qt.io/archives/qt-4.8/qt.html#PenStyle-enum
    #         for line types. If None (default), user defined color scheme will be used.
    #     """
    #
    #     for sp in self:
    #         sp.set_style(color, line_width, line_type, False)
    #     if redraw_spectra:
    #         self._redraw_all_spectra()

    #
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

        for sp in self:
            idx = find_nearest_idx(sp.data[:, 0], x)
            ret_list.append(sp.data[idx, 1])

        return np.asarray(ret_list, dtype=np.float64)

    def get_transpose(self, max_items=1000):
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

        x, y, matrix = group2mat(self.__iter__())

        if y is None:
            raise ValueError("Names of spectra cannot be parsed to float.")

        n = x.shape[0]
        assert n == matrix.shape[0]

        group_name = "Transpose of {}".format(self.name)
        spectra = []

        if n > max_items:
            raise ValueError(
                "Number of transposed items ({}) exceeded the maximum number ({}). Cannot transpose spectra.".format(
                    n, max_items))

        for i in range(n):
            sp = Spectrum.from_xy_values(y, matrix[i], name=str(x[i]))
            spectra.append(sp)

        return SpectrumList(spectra, name=group_name)

        # self.add_to_list([spectra])

    def _arithmetic_operation(self, other, func_operation):
        """
        Perform an defined operation on groups of spectra (type Spectrum). func_operation is a pointer to function.
        This function takes 2 arguments that must be type of Spectrum.
        """
        # operation with another list, group + group
        if isinstance(other, (SpectrumList, np.ndarray)):
            if len(self) != len(other):
                raise ValueError("Cannot perform an operation on groups which contains different number of items.")
            if len(self) == 0:
                return SpectrumList()
            # if not isinstance(self[0], Spectrum) or not isinstance(other[0], Spectrum):
            #     raise ValueError("Objects in list have to be type of Spectrum.")
            ret_list = SpectrumList()
            for i in range(len(self)):
                ret_list.children.append(func_operation(self[i], other[i]))
            return ret_list, other.name if isinstance(other, SpectrumList) else 'ndarray'

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
