
import numpy as np
# import math
from abc import abstractmethod

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq
from scipy.integrate import cumtrapz, simps
from scipy.fftpack import dct, idct
from numpy.linalg import norm

from typing import Union, Iterable, List


import functools

# from https://stackoverflow.com/questions/40104377/issiue-with-implementation-of-2d-discrete-cosine-transform-in-python
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


## from https://github.com/Tillsten/skultrafast/blob/master/skultrafast/dv.py
# TODO simlify, benchmark
def fi(array: np.ndarray, values: Union[int, float, Iterable]) -> Union[int, List[int]]:
    """
    Finds index of nearest `value` in `array`. If value >  max(array), the last index of array
    is returned, if value < min(array), 0 is returned. Array must be sorted. Also works for value
    to be array, then array of indexes is returned.

    Parameters
    ----------
    array : ndarray
        Array to be searched.
    values : {int, float, list}
        Value or values to look for.

    Returns
    -------
    out : int, np.ndarray
        Found nearest index/es to value/s.
    """
    if not np.iterable(values):
        values = [values]

    ret_idx = [np.argmin(np.abs(array - i)) for i in values]

    return ret_idx[0] if len(ret_idx) == 1 else ret_idx


# def find_nearest(array, value):
#     """
#     Returns closest value in in `array`. If value >  max(array), the last value of array
#     is returned, if value < min(array), first value of array is returned. Array must be sorted.
#
#     Parameters
#     ----------
#     array : ndarray
#         Array to be searched.
#     value : {int, float}
#         Value.
#
#     Returns
#     -------
#     out : type of values in `array`
#         Found nearest value.
#     """
#     idx = fi(array, value)
#     return array[idx]


def group2mat(spectra):
    """Given a generator of spectra (SpectrumList, list, etc.), it returns the x values, 'name'
     values converted to float and value matrix in shape of [x, y]. All spectra has to have in
     same shape. If some names cannot be converted to float, None is returned for y values.

    Parameters
    ----------
    spectra : iterable
        Generator of spectra (SpectrumList, list, etc.)

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


def operation(operator=lambda a, b: a + b, operator_str='+', switch_names=False):
    """TODO--->"""
    def decorator(fn):
        @functools.wraps(fn)
        def func(self, other):
            obj, other_str = self._arithmetic_operation(other, operator)
            if obj is None:
                return NotImplemented
            if switch_names:
                obj.name = f"{other_str} {operator_str} {self.name}"
            else:
                obj.name = f"{self.name} {operator_str} {other_str}"

            return obj
        return func
    return decorator


class IOperationBase:
    """
    Interface that defines the operations for Spectrum and SpectrumList objects.
    It allows for ... TODO...
    """

    def __init__(self, name):
        self.name = name

    def _redraw_all_spectra(self):
        pass

    def _update_view(self):
        pass

    @abstractmethod
    def _arithmetic_operation(self, other, operator):
        """TODO--->>"""
        pass

    @operation(lambda a, b: a + b, '+', False)  # self + other
    def __add__(self, other):
        pass

    @operation(lambda a, b: a - b, '-', False)  # self - other
    def __sub__(self, other):
        pass

    @operation(lambda a, b: a * b, '*', False)  # self * other
    def __mul__(self, other):
        pass

    @operation(lambda a, b: a / b, '/', False)  # self / other
    def __truediv__(self, other):
        pass

    @operation(lambda a, b: b + a, '+', True)  # other + self, switch_names=True
    def __radd__(self, other):
        pass

    @operation(lambda a, b: b - a, '-', True)  # other - self, switch_names=True
    def __rsub__(self, other):
        pass

    @operation(lambda a, b: b * a, '*', True)  # other * self, switch_names=True
    def __rmul__(self, other):
        pass

    @operation(lambda a, b: b / a, '/', True)  # other / self, switch_names=True
    def __rtruediv__(self, other):
        pass

    @operation(lambda a, b: a ** b, '**', False)  # self ** power
    def __pow__(self, power, modulo=None):
        pass

    @operation(lambda a, b: b ** a, '**', True)  # other ** self, switch_names=True
    def __rpow__(self, other):
        pass

    def __neg__(self):  # - self
        obj = self * -1
        obj.name = f'-{self.name}'
        return obj

    def __pos__(self):  # + self
        return self.__copy__()

    def __copy__(self):
        return NotImplementedError


class SpectrumList(IOperationBase):

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

        super(SpectrumList, self).__init__(name)
        self.children = [] if children is None else children

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
            idx = fi(sp.data[:, 0], x)
            ret_list.append(sp.data[idx, 1])

        return np.asarray(ret_list, dtype=np.float64)

    def get_sum(self):
        if len(self) < 2:
            raise ValueError("At least 2 items have to be in the group in order to perform transposition.")

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        x, y, matrix = group2mat(self.__iter__())

        return Spectrum.from_xy_values(x, matrix.sum(axis=1), name=f'{self.name}-sum')

    def get_average(self):
        if len(self) < 2:
            raise ValueError("At least 2 items have to be in the group in order to perform transposition.")

        if not isinstance(self[0], Spectrum):
            raise ValueError("Objects in list have to be type of Spectrum.")

        x, y, matrix = group2mat(self.__iter__())

        return Spectrum.from_xy_values(x, matrix.mean(axis=1), name=f'{self.name}-avrg')

    def get_transpose(self, max_items=1000):
        """
        Transposes group, names of spectra in group will be taken as x values for transposed data, so these values
        must be convertible to int or float numbers. No text is allowed in these cells, only values.
        All x values of spectra in the group will become names in new group. The transposed group is added to Tree Widget.
        The operation is basically the same as copying the group of spectra to Excel, performs transposition of the matrix
        and copying back to spectramanipulator.

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
        elif isinstance(other, (Spectrum, float, int)):
            if len(self) == 0:
                return SpectrumList()

            if not isinstance(self[0], Spectrum):
                raise ValueError("Objects in list have to be type of Spectrum.")
            ret_list = SpectrumList()
            for sp in self:
                ret_list.children.append(func_operation(sp, other))
            return ret_list, str(other) if isinstance(other, (float, int)) else other.name

        return None, None

    def __neg__(self):
        sl = SpectrumList()
        sl.name = f'-{self.name}'
        for sp in self:
            sl.children.append(-sp)
        return sl

    def __getitem__(self, item):
        return self.children[item]

    def __iter__(self):
        return iter(self.children)

    def __repr__(self):
        if self.__len__() == 0:  # self.__class__.__name__ gets the name of the current class instance
            return f'{self.__class__.__name__} without data'
        return f"{self.__class__.__name__}({self.name}, {self.__len__()} items)"

    def __copy__(self):
        """Deep copy this instance as SpectrumList."""
        ret = SpectrumList(name=self.name)
        for child in self.children:
            ret.children.append(child.__copy__())
        return ret


def add_modif_func(redraw_spectra=True, update_view=False):
    """Adds the function to cls
    adds a new functions with """
    # https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class

    class Decorator:
        def __init__(self, fn):
            self.fn = fn

        def __set_name__(self, owner, fn_name):
            # gets called when the owner class is created !!

            # do something with owner, i.e.
            # print(f"decorating {self.fn} and using {owner}")
            # self.fn.class_name = owner.__name__

            @functools.wraps(self.fn)
            def fn_spectrum(this, *args, **kwargs):
                ret_vals = self.fn(this, *args, **kwargs)

                if redraw_spectra:
                    this._redraw_all_spectra()
                if update_view:
                    this._update_view()

                return ret_vals

            @functools.wraps(self.fn)
            def fn_spectrum_list_no_update(this, *args, **kwargs):
                for sp in this:
                    self.fn(sp, *args, **kwargs)  # perform the operation for each spectrum

                return this

            @functools.wraps(self.fn)
            def fn_spectrum_list(this, *args, **kwargs):
                for sp in this:
                    self.fn(sp, *args, **kwargs)  # perform the operation for each spectrum

                if redraw_spectra:
                    this._redraw_all_spectra()
                if update_view:
                    this._update_view()

                return this

            # then replace ourself with the original method
            setattr(owner, f'{fn_name}_no_update', self.fn)  # no update as with original function
            setattr(owner, fn_name, fn_spectrum)  # use same name for modified fcn
            setattr(SpectrumList, f'{fn_name}_no_update', fn_spectrum_list_no_update)
            setattr(SpectrumList, fn_name, fn_spectrum_list)

    return Decorator


def add_op_func():

    def decorator(fn):
        @functools.wraps(fn)
        def fn_spectrum_list(self, *args, **kwargs):
            # perform the operation for each spectrum and collect the results as ndarray
            return np.asarray([fn(sp, *args, **kwargs) for sp in self])

        setattr(SpectrumList, fn.__name__, fn_spectrum_list)

        return fn

    return decorator


class Spectrum(IOperationBase):

    def __init__(self, data=None, name='', filepath=None, assume_sorted=False, **kwargs):
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

        super(Spectrum, self).__init__(name)

        if assume_sorted:
            self.data = np.asarray(data, dtype=np.float64) if data is not None else None
        else:
            # sort according to first column,numpy matrix, 1. column wavelength, 2. column absorbance
            self.data = np.asarray(data[data[:, 0].argsort()], dtype=np.float64) if data is not None else None

        self.filepath = filepath

    @classmethod
    def from_xy_values(cls, x_values, y_values, name='', filepath=None, **kwargs):
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

        # check whether the x and y values can be iterated over
        if not np.iterable(x_values) and not np.iterable(y_values):
            raise ValueError(
                "Argument error, x_values and y_values must be iterables and contain numbers.")

        try:
            if len(x_values) != len(y_values):
                raise ValueError("Length of x_values and y_values must match.")

            x_data = np.asarray(x_values, dtype=np.float64)
            y_data = np.asarray(y_values, dtype=np.float64)
        except ValueError:
            raise

        data = np.vstack((x_data, y_data)).T

        return cls(data=data, name=name, filepath=filepath, assume_sorted=False, **kwargs)

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

    @add_modif_func(True, True)
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

    @add_modif_func(True, False)
    def whittaker(self, lam: float = 1e3):
        """
        Applies Whittaker smoother to a spectrum. Based on 10.1016/j.csda.2009.09.020, utilizes discrete cosine
        transform to efficiently perform the calculation.

        Correctly smooths only evenly spaced data!!

        Parameters
        ----------
        lam : float
            Lambda - parametrizes the roughness of the smoothed curve.
        """

        N = self.data.shape[0]

        Lambda = -2 + 2 * np.cos(np.arange(N) * np.pi / N)  # eigenvalues of 2nd order difference matrix

        gamma = 1 / (1 + lam * Lambda * Lambda)
        self.data[:, 1] = idct(gamma * dct(self.data[:, 1], norm='ortho'), norm='ortho')

        return self



    @add_modif_func(True, False)
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

    @add_modif_func(True, False)
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

    @add_modif_func(True, True)
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

    @add_modif_func(True, False)
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

    def _get_start_end_indexes(self, x0=None, x1=None):
        start = 0
        end = self.data.shape[0]

        if x0 is not None and x1 is not None and x0 > x1:
            x0, x1 = x1, x0

        if x0 is not None:
            start = fi(self.data[:, 0], x0)

        if x1 is not None:
            end = fi(self.data[:, 0], x1) + 1

        return start, end

    def _get_xy_from_range(self, x0=None, x1=None):
        start, end = self._get_start_end_indexes(x0, x1)

        x = self.data[start:end, 0]
        y = self.data[start:end, 1]

        return x, y

    @add_op_func()
    def integral(self, x0=None, x1=None, method: str = 'trapz'):
        """Calculate integral of the spectrum at specified x range [x0, x1] by integration method.
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
            return (y[:-1] * x_diff).sum()
        elif method == 'trapz':
            return np.trapz(y, x)
        else:  # simpsons rule
            return simps(y, x)

    @add_modif_func(True, False)
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

    @add_modif_func(True, False)
    def baseline_corr_arPLS(self, lam: float = 1e3, niter: int = 100, tol: float = 2e-3):
        """
        Performs baseline correction using asymmetrically reweighted penalized least squares (arPLS). Based on
        10.1016/j.csda.2009.09.020 and 10.1039/c4an01061b, utilizes discrete cosine transform to efficiently
        perform the calculation.

        Correctly smooths only evenly spaced data!!

        Parameters
        ----------
        lam : float
            Lambda - parametrizes the roughness of the smoothed curve.
        niter : int
            Maximum number of iterations.
        tol: float
            Tolerance for convergence based on weight matrix.
        """

        N = self.data.shape[0]

        Lambda = -2 + 2 * np.cos(np.arange(N) * np.pi / N)  # eigenvalues of 2nd order difference matrix

        gamma = 1 / (1 + lam * Lambda * Lambda)

        y_orig = self.data[:, 1].copy()
        z = y_orig  # initialize baseline
        y_corr = None  # data corrected for baseline
        w = np.ones_like(z)  # weight vector

        i = 0
        crit = 1

        while crit > tol and i < niter:
            z = idct(gamma * dct(w * (y_orig - z) + z, norm='ortho'), norm='ortho')  # calculate the baseline

            y_corr = y_orig - z  # data corrected for baseline
            y_corr_neg = y_corr[y_corr < 0]  # negative data values

            m = np.mean(y_corr_neg)
            s = np.std(y_corr_neg)

            new_w = 1 / (1 + np.exp(2 * (y_corr - (2 * s - m)) / s))  # update weights with logistic function

            crit = norm(new_w - w) / norm(new_w)
            w = new_w

            if (i + 1) % int(np.sqrt(niter)) == 0:
                print(f'Iteration={i + 1}, {crit=:.2g}')
            i += 1

        self.data[:, 1] = y_corr

        return self, Spectrum.from_xy_values(self.data[:, 0], z, f'{self.name} - baseline'),\
               Spectrum.from_xy_values(self.data[:, 0], y_orig, f'{self.name} - original data')   # return the corrected data, baseline and the original data

    @add_modif_func(True, False)
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

    @add_op_func()
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

    @add_modif_func(True, True)
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
            raise ValueError(f"Spacing ({spacing}) cannot be larger that data itself.")

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

    @add_modif_func(True, True)
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

    @add_modif_func(True, False)
    def ceil_larger_to(self, value: float, x0=None, x1=None):
        """Values larger than value in the range [x0, x1] will be assigned to value .

        Parameters
        ----------
        value : {int, float}
            Value to ceil to.
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """

        start, end = self._get_start_end_indexes(x0, x1)

        y = self.data[start:end, 1]
        y[y > value] = value

        self.data[start:end, 1] = y

        return self

    @add_modif_func(True, False)
    def floor_lower_to(self, value: float, x0=None, x1=None):
        """Values lower than value in the range [x0, x1] will be assigned to a value .

        Parameters
        ----------
        value : {int, float}
            Value to floor to.
        x0 : {int, float}
            First x value.
        x1 : {int, float}
            Last x value.
        """

        start, end = self._get_start_end_indexes(x0, x1)

        y = self.data[start:end, 1]
        y[y < value] = value

        self.data[start:end, 1] = y

        return self

    @add_modif_func(True, True)
    def extend_by_value(self, x0=None, x1=None, value=0):
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
            start_idx = fi(self.data[:, 0], x0)

        if x1 is not None:
            end_idx = fi(self.data[:, 0], x1) + 1

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

            min_stack = np.ones((2, num_min - 1)) * value
            min_stack[0] = min_lin_space[:-1]
            min_stack = min_stack.T
        except ValueError:
            pass

        max_stack = None
        try:
            new_x_max = spacing * int(np.round(x1 / spacing, 0))

            num_max = int((new_x_max - x_max) / spacing + 1)
            max_lin_space = np.linspace(x_max, new_x_max, num=num_max)

            max_stack = np.ones((2, num_max - 1)) * value
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
        if not isinstance(other, (Spectrum, int, float, list, tuple, np.ndarray)):
            return None, None

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
            other_str = repr(other)

        else:  # number
            y_data = func_operation(self.data[:, 1], other)
            other_str = repr(other)

        max_len = 20
        if len(other_str) > max_len:
            other_str = other_str[:max_len]

        return Spectrum.from_xy_values(self.data[:, 0], np.nan_to_num(y_data)), other_str

    def __repr__(self):
        if self.data is None:
            return f'{self.__class__.__name__} without data'
        return f"{self.__class__.__name__}({self.name}, x_range=({self.data[0, 0]:.4g}, {self.data[-1, 0]:.4g}), " \
               f"n_points={self.data.shape[0]})"

    def __copy__(self):
        """Deep copy the current instance as Spectrum object."""
        return Spectrum(self.data.copy(), filepath=self.filepath, name=self.name, assume_sorted=True)


