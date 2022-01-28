import numpy as np
from typing import Iterable
import os

import matplotlib as mpl
import matplotlib.pyplot as plt  # we plot graphs with this library
from matplotlib import cm
from matplotlib.ticker import *
import matplotlib.gridspec as gridspec

from matplotlib import colors as c

# from copy import deepcopy
from PyQt5.QtWidgets import QApplication
from scipy.linalg import lstsq

from spectramanipulator.settings.settings import Settings
from spectramanipulator.spectrum import fi, Spectrum, SpectrumList, group2mat
from scipy.integrate import simps, cumtrapz
from scipy.stats import linregress
from uncertainties import ufloat, unumpy


import functools

# for backward compatibility of smpj files
ItemList = list

WL_LABEL = 'Wavelength / nm'
WN_LABEL = "Wavenumber / $10^{4}$ cm$^{-1}$"


# needed for correctly display tics for symlog scale
class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]  # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper e	nd

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (
                dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] * 10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] - self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (
                dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1] * 10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1] + self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


def setup_wavenumber_axis(ax, x_label=WN_LABEL,
                          x_major_locator=None, x_minor_locator=AutoMinorLocator(5), factor=1e3):
    secondary_ax = ax.secondary_xaxis('top', functions=(lambda x: factor / x, lambda x: 1 / (factor * x)))

    secondary_ax.tick_params(which='major', direction='in')
    secondary_ax.tick_params(which='minor', direction='in')

    if x_major_locator:
        secondary_ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        secondary_ax.xaxis.set_minor_locator(x_minor_locator)

    secondary_ax.set_xlabel(x_label)

    return secondary_ax


def set_main_axis(ax, x_label=WL_LABEL, y_label="Absorbance", xlim=(None, None), ylim=(None, None),
                  x_major_locator=None, x_minor_locator=None, y_major_locator=None, y_minor_locator=None):
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if xlim[0] is not None:
        ax.set_xlim(xlim)
    if ylim[0] is not None:
        ax.set_ylim(ylim)

    if x_major_locator:
        ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        ax.xaxis.set_minor_locator(x_minor_locator)

    if y_major_locator:
        ax.yaxis.set_major_locator(y_major_locator)

    if y_minor_locator:
        ax.yaxis.set_minor_locator(y_minor_locator)

    ax.tick_params(axis='both', which='major', direction='in')
    ax.tick_params(axis='both', which='minor', direction='in')


def _transform_func(transform=lambda y: y):
    def decorator(fn):
        @functools.wraps(fn)
        def func(item):
            fn_name = fn.__name__
            if isinstance(item, Spectrum):
                y_data = transform(item.data[:, 1])
                return Spectrum.from_xy_values(item.data[:, 0], np.nan_to_num(y_data), name=f'{fn_name}({item.name})')
            elif isinstance(item, SpectrumList):
                sl = SpectrumList(name=f'{fn_name}({item.name})')
                for sp in item:
                    y_data = transform(sp.data[:, 1])
                    new_sp = Spectrum.from_xy_values(sp.data[:, 0], np.nan_to_num(y_data), name=f'{fn_name}({sp.name})')
                    sl.children.append(new_sp)
                return sl
            else:
                return transform(item)  # this may not be implemented, but for numbers and ndarrays it will work
        return func
    return decorator


@_transform_func(lambda y: np.exp(y))
def exp(item):
    """
    Calculates the exponential of y values and returns a new Spectrum/SpectrumList.
    """
    pass


@_transform_func(lambda y: np.log10(y))
def log10(item):
    """
    Calculates the decadic logarithm of y values and returns a new Spectrum/SpectrumList.
    """
    pass


@_transform_func(lambda y: np.log(y))
def log(item):
    """
    Calculates the natural logarithm of y values and returns a new Spectrum/SpectrumList
    """
    pass


@_transform_func(lambda y: -np.log10(-y))
def T2A_LFP(item):
    """
    Performs the transmittance to absorbance conversion for nano kinetics,
    y_transformed = -log10(-y)."""
    pass

#
# def add_to_list(spectra):
#     """
#     Copies all spectra and imports them to the Tree Widget.
#
#     Parameters
#     ----------
#     spectra : {:class:`Spectrum`, :class:`SpectrumList`, list, list of lists, SpectrumItemGroup, SpectrumItem}
#         The input spectra to be added into Tree Widget.
#     """
#
#     if UserNamespace.instance is not None:
#         UserNamespace.instance.tw.add_to_list(spectra)


# def load_kinetics(spectra):
#     """
#     Copies all spectra and imports them to the Tree Widget.
#
#     Parameters
#     ----------
#     spectra : {:class:`Spectrum`, :class:`SpectrumList`, list, list of lists, SpectrumItemGroup, SpectrumItem}
#         The input spectra to be added into Tree Widget.
#     """
#
#     if UserNamespace.instance is not None:
#         UserNamespace.instance.tw.add_to_list(spectra)



def import_files(filepaths):
    """
    Imports the filepaths and add to Tree Widget

    Parameters
    ----------
    filepaths : list of strs or str
        List of filepaths to import.
    """
    if UserNamespace.instance is not None:
        UserNamespace.instance.main.tree_widget.import_files(filepaths)


def set_xy_range(x0=None, x1=None, y0=None, y1=None, padding=0):
    """
    Changes the x and y ranges of scene in Plot Widget.

    Parameters
    ----------
    x0 : {int, float, None}
        New fist x value. If None, old value is kept.
    x1 : {int, float, None}
        New last x value. If None, old value is kept.
    y0 : {int, float, None}
        New fist y value. If None, old value is kept.
    y1 : {int, float, None}
        New last y value. If None, old value is kept.
    padding : {int, float}
        Sets the padding around the choosed rectangle. If 0, no padding will be used.
    """
    plot_widget = UserNamespace.instance.main.grpView

    x_range, y_range = plot_widget.plotItem.getViewBox().viewRange()

    plot_widget.plotItem.getViewBox().setXRange(x_range[0] if x0 is None else x0,
                                                x_range[1] if x1 is None else x1,
                                                padding=padding)

    plot_widget.plotItem.getViewBox().setYRange(y_range[0] if y0 is None else y0,
                                                y_range[1] if y1 is None else y1,
                                                padding=padding)


def set_default_HSV_color_scheme():
    """Sets the default values for HSV color scheme."""
    Settings.hues = 9
    Settings.values = 1
    Settings.maxValue = 255
    Settings.minValue = 150
    Settings.maxHue = 360
    Settings.minHue = 0
    Settings.sat = 255
    Settings.alpha = 255

    if Settings.HSV_color_scheme:
        redraw_all_spectra()


def set_HSV_color_scheme(active=True, **kwargs):
    """Set the options for HSV color scheme and whether the scheme is active.

    Options
    -------
    ================  =================================================================================
    *active* (bool)   True for setting the scheme active, False for not (default color scheme will be
                      used).
    *hues* (int)      The number of hues that will be repeating, default 9.
    *values* (int)    The number of values/brightnesses that will be repeating, default 1.
    *minValue* (int)  A minimum value/brightness, this can be <0, 255>, default 150.
    *maxValue* (int)  A maximum value/brightness, this can be <0, 255>, default 255.
    *minHue* (int)    A minimum hue, this can be <0, 360>, default 0
    *maxHue* (int)    A maximum hue, this can be <0, 360>, default 360
    *sat* (int)       The saturation value, this can be <0, 255>, default 255
    *alpha* (int)     The transparency value, this can be <0, 255>, default 255
    ================  =================================================================================
    """
    hues = kwargs.get('hues', None)
    values = kwargs.get('values', None)
    maxValue = kwargs.get('maxValue', None)
    minValue = kwargs.get('minValue', None)
    maxHue = kwargs.get('maxHue', None)
    minHue = kwargs.get('minHue', None)
    sat = kwargs.get('sat', None)
    alpha = kwargs.get('alpha', None)

    Settings.HSV_color_scheme = active
    Settings.hues = hues if hues is not None else Settings.hues
    Settings.values = values if values is not None else Settings.values
    Settings.maxValue = maxValue if maxValue is not None else Settings.maxValue
    Settings.minValue = minValue if minValue is not None else Settings.minValue
    Settings.maxHue = maxHue if maxHue is not None else Settings.maxHue
    Settings.minHue = minHue if minHue is not None else Settings.minHue
    Settings.sat = sat if sat is not None else Settings.sat
    Settings.alpha = alpha if alpha is not None else Settings.alpha

    redraw_all_spectra()


def copy_to_clipboard(array, delimiter='\t', decimal_sep='.', new_line='\n'):
    """Copies the *array* of numbers into clipboard. This can be then pasted to Excel for example.

    Parameters
    ----------
    array : {array_like, iterable}
        Array of values. Can be 1D or 2D array
    delimiter : str
        Delimiter between numbers, default tabulator '\\\\t'
    decimal_sep : str
        Decimal separator, default '.'
    new_line : str
        New line character, default '\\\\n'
    """
    if not isinstance(array, (np.ndarray, Iterable, list, tuple)):
        raise ValueError(f"Cannot copy {type(array)} to clipboard.")

    try:
        text = new_line.join(delimiter.join(str(num).replace('.', decimal_sep) for num in row) for row in array)
    except:  # the second dimension is not iterable, we probably got only 1D array, so lets put into clipboard only this
        text = delimiter.join(str(num).replace('.', decimal_sep) for num in array)

    cb = QApplication.clipboard()
    cb.clear(mode=cb.Clipboard)
    cb.setText(text, mode=cb.Clipboard)


def update_view():
    """Updates the Tree Widget."""
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main

    mw.tree_widget.update_view()
    mw.tree_widget.setup_info()


def redraw_all_spectra():
    """Redraws all spectra."""
    if UserNamespace.instance is None:
        return

    mw = UserNamespace.instance.main

    mw.redraw_all_spectra()

###  Calculation of epsilon from concentration-dependent absorption spectra, the name of the spectra must contain
###  real concentration, the spectra must me ordered from lowest to highest concentration

def _get_C(group):
    """Returns parsed names to floats from a group"""
    x_vals_temp = []
    for sp in group:
        try:
            x_vals_temp.append(float(sp.name.replace(',', '.').strip()))
        except ValueError:
            raise ValueError("Names of spectra cannot be parsed to float.")
    return np.asarray(x_vals_temp, dtype=np.float64)

# def _get_D(group):
#     D = group[0].data[:, 1]
#     for i in range(1, len(group)):
#         D = np.vstack((D, group[i].data[:, 1]))
#     return D


def calc_Eps(group):

    wls, c, D = group2mat(group)

    # C needs to be changed to column vector
    ST = lstsq(c[:, None], D.T)[0]


    # add a spectrum to list
    return Spectrum.from_xy_values(wls, ST.flatten(), name=group.name + '-epsilon')


def rename_times(group, decimal_places=1):
    """Renames the group that has names in seconds. Changes for minutes for 60s <= time < 1 hour to minutes and
    time >= 1 hour to hours."""

    parsed_times = []
    times = _get_C(group)

    for time in times:
        unit = ' s'
        if time >= 3600:
            time /= 3600
            unit = ' h'
        elif 60 <= time < 3600:
            time /= 60
            unit = ' min'

        time = np.round(time, decimal_places)
        parsed_times.append(f'{time}{unit}')

    group.set_names(parsed_times)


# def load_kinetics(dir_name, spectra_dir_name='spectra', times_fname='times.txt', blank_spectrum='blank.dx', dt=None,
#                   b_corr=None, cut=None, corr_to_zero_time=True):
#     """Given a directory name that contains folders of individual experiments, it loads all kinetics.
#        each experiment folder must contain folder spectra (or defined in spectra_dir_name arg.)
#         if blank is given, it will be subtracted from all spectra, times.txt will contain
#         times for all spectra, optional baseline correction and cut can be done.
#
#     Folder structure:
#         [dir_name]
#             [exp1_dir]
#                 [spectra]
#                     01.dx (or .csv or .txt)
#                     02.dx
#                     ...
#                 times.txt (optional)
#                 blank.dx (optional)
#             [exp2_dir]
#                 ...
#             ...
#     """
#
#     if UserNamespace.instance is None:
#         return
#
#     if not os.path.isdir(dir_name):
#         raise ValueError(f'{dir_name}  does not exist!')
#
#     for item in os.listdir(dir_name):
#         path = os.path.join(dir_name, item)
#         if not os.path.isdir(path):
#             continue
#
#         load_kinetic(path, spectra_dir_name=spectra_dir_name, times_fname=times_fname, blank_spectrum=blank_spectrum,
#                      dt=dt, b_corr=b_corr, cut=cut, corr_to_zero_time=corr_to_zero_time)
#
#
# def load_kinetic(dir_name, spectra_dir_name='spectra', times_fname='times.txt', blank_spectrum='blank.dx', dt=None,
#                   b_corr=None, cut=None, corr_to_zero_time=True):
#     """Given a directory name, it loads all spectra in dir named "spectra" - func. arg.,
#     if blank is given, it will be subtracted from all spectra, times.txt will contain
#     times for all spectra, optional baseline correction and cut can be done.
#
#     Folder structure:
#         [dir_name]
#             [spectra]
#                 01.dx
#                 02.dx
#                 ...
#             times.txt (optional)
#             blank.dx (optional)
#     """
#
#     if UserNamespace.instance is None:
#         return
#
#     tw = UserNamespace.instance.main.tree_widget
#     root = tw.myModel.root  # item in IPython console
#
#     if not os.path.isdir(dir_name):
#         raise ValueError(f'{dir_name}  does not exist!')
#
#     spectra_path = os.path.join(dir_name, spectra_dir_name)
#
#     if not os.path.isdir(spectra_path):
#         raise ValueError(f'{spectra_dir_name}  does not exist in {dir_name}!')
#
#     spectras = [os.path.join(spectra_path, filename) for filename in os.listdir(spectra_path)]
#
#     n_items_before = root.__len__()
#     tw.import_files(spectras)
#     n_spectra = root.__len__() - n_items_before
#
#     tw.add_items_to_group(root[n_items_before:], edit=False)  # add loaded spectra to group
#     root[n_items_before].name = f'raw [{os.path.split(dir_name)[1]}]'  # set name of a group
#
#     times = np.asarray([dt * i for i in range(n_spectra)]) if dt is not None else None
#     # idx_add = 0
#     group_idx = n_items_before
#     blank_used = False
#
#     # load explicit times
#     times_fpath = os.path.join(dir_name, times_fname)
#     if os.path.isfile(times_fpath):
#         tw.import_files(times_fpath)
#         # idx_add += 1
#         if times is None:
#             times = root[-1].data[:, 0].copy()
#             if corr_to_zero_time:
#                 times -= times[0]
#
#         # push times variable to the console
#         UserNamespace.instance.main.console.push_variables(
#             {
#                 'times': times
#             }
#         )
#
#     if times is not None:
#         root[group_idx].set_names(times)
#
#     # load blank spectrum if available
#     blank_fpath = os.path.join(dir_name, blank_spectrum)
#     if os.path.isfile(blank_fpath):
#         last_idx = root.__len__() - 1
#         tw.import_files(blank_fpath)
#         add_to_list(root[group_idx] - root[last_idx + 1])
#         if times is not None:
#             root[-1].set_names(times)
#         blank_used = True
#
#     corr_idx = -1 if blank_used else group_idx
#
#     if b_corr is not None:
#         root[corr_idx].baseline_correct(*b_corr)
#         root[corr_idx].name += 'bcorr'
#     if cut is not None:
#         root[corr_idx].cut(*cut)
#         root[corr_idx].name += 'cut'
#
#     # return times
#

def _setup_wavenumber_axis(ax, x_label=WN_LABEL,
                          x_major_locator=None, x_minor_locator=AutoMinorLocator(5), factor=1e3):
    secondary_ax = ax.secondary_xaxis('top', functions=(lambda x: factor / x, lambda x: 1 / (factor * x)))

    secondary_ax.tick_params(which='major', direction='in')
    secondary_ax.tick_params(which='minor', direction='in')

    if x_major_locator:
        secondary_ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        secondary_ax.xaxis.set_minor_locator(x_minor_locator)

    secondary_ax.set_xlabel(x_label)

    return secondary_ax


def _set_main_axis(ax, x_label=WL_LABEL, y_label="Absorbance", xlim=(None, None), ylim=(None, None),
                  x_major_locator=None, x_minor_locator=None, y_major_locator=None, y_minor_locator=None,
                   direction='in'):
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if xlim[0] is not None:
        ax.set_xlim(xlim)
    if ylim[0] is not None:
        ax.set_ylim(ylim)

    if x_major_locator:
        ax.xaxis.set_major_locator(x_major_locator)

    if x_minor_locator:
        ax.xaxis.set_minor_locator(x_minor_locator)

    if y_major_locator:
        ax.yaxis.set_major_locator(y_major_locator)

    if y_minor_locator:
        ax.yaxis.set_minor_locator(y_minor_locator)

    ax.tick_params(axis='both', which='major', direction=direction)
    ax.tick_params(axis='both', which='minor', direction=direction)


def setup_twin_x_axis(ax, y_label="$I_{0,\\mathrm{m}}$ / $10^{-10}$ einstein s$^{-1}$ nm$^{-1}$",
                      x_label=None, ylim=(None, None), y_major_locator=None, y_minor_locator=None,
                      keep_zero_aligned=True):
    ax2 = ax.twinx()

    ax2.tick_params(which='major', direction='in')
    ax2.tick_params(which='minor', direction='in')

    if y_major_locator:
        ax2.yaxis.set_major_locator(y_major_locator)

    if y_minor_locator:
        ax2.yaxis.set_minor_locator(y_minor_locator)

    ax2.set_ylabel(y_label)

    if keep_zero_aligned and ylim[0] is None and ylim[1] is not None:
        # a = bx/(x-1)
        ax1_ylim = ax.get_ylim()
        x = -ax1_ylim[0] / (ax1_ylim[1] - ax1_ylim[0])  # position of zero in ax1, from 0, to 1
        a = ylim[1] * x / (x - 1)  # calculates the ylim[0] so that zero position is the same for both axes
        ax2.set_ylim(a, ylim[1])

    elif ylim[0] is not None:
        ax2.set_ylim(ylim)

    return ax2


def plot_kinetics(kin_group_items: list, n_rows: int = None, n_cols: int = None, n_spectra=50, linscale=1,
                 linthresh=100, cmap='jet_r', major_ticks_labels=(100, 1000), emph_t=(0, 200, 1000),
                 inset_loc=(0.75, 0.1, 0.03, 0.8), colorbar_label='Time / s', lw=0.5, alpha=0.5,
                 fig_size_one_graph=(5, 4), y_label='Absorbance', x_label=WL_LABEL, x_lim=(230, 600), filepath=None,
                 dpi=500, transparent=True, LED_sources: list = None):

    kin_group_items = kin_group_items if isinstance(kin_group_items, list) else [kin_group_items]
    n = len(kin_group_items)  # number of EEMs to plot

    if LED_sources is not None:
        LED_sources = LED_sources if isinstance(LED_sources, list) else [LED_sources]
        if len(LED_sources) == 1 and n > 1:
            LED_sources = LED_sources * n

        assert len(LED_sources) == n,  "Number of provided LEDs must be the same as spectra"
    else:
        LED_sources = [None] * n

    if n_rows is None and n_cols is None:  # estimate the n_rows and n_cols from the sqrt of number of graphs
        sqrt = n ** 0.5
        n_rows = int(np.ceil(sqrt))
        n_cols = int(sqrt)
    elif n_rows is None and n_cols is not None:
        n_rows = int(np.ceil(n / n_cols))
    elif n_rows is not None and n_cols is None:
        n_cols = int(np.ceil(n / n_rows))

    # assert n_rows * n_cols >= n  # not necessary, if the condition is not valid, fewer plots will be plotted

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size_one_graph[0] * n_cols, fig_size_one_graph[1] * n_rows))

    axes = axes.flatten() if np.iterable(axes) else [axes]

    for ax, group, LED_source in zip(axes, kin_group_items, LED_sources):

        t = np.asarray(group.get_names(), dtype=np.float64)
        w = group[0].data[:, 0]

        _set_main_axis(ax, x_label=x_label, y_label=y_label, xlim=x_lim, x_minor_locator=None, y_minor_locator=None)
        _ = _setup_wavenumber_axis(ax)

        cmap = cm.get_cmap(cmap)
        norm = mpl.colors.SymLogNorm(vmin=t[0], vmax=t[-1], linscale=linscale, linthresh=linthresh, base=10, clip=True)

        tsb_idxs = fi(t, emph_t)
        ts_real = np.round(t[tsb_idxs])

        x_space = np.linspace(0, 1, n_spectra, endpoint=True, dtype=np.float64)

        t_idx_space = fi(t, norm.inverse(x_space))
        t_idx_space = np.sort(np.asarray(list(set(t_idx_space).union(set(tsb_idxs)))))

        for i in t_idx_space:
            x_real = norm(t[i])
            x_real = 0 if np.ma.is_masked(x_real) else x_real
            ax.plot(w, group[i].data[:, 1], color=cmap(x_real),
                     lw=1.5 if i in tsb_idxs else lw,
                     alpha=1 if i in tsb_idxs else alpha,
                     zorder=1 if i in tsb_idxs else 0)

        cbaxes = ax.inset_axes(inset_loc)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbaxes, orientation='vertical',
                            format=mpl.ticker.ScalarFormatter(),
                            label=colorbar_label)

        cbaxes.invert_yaxis()

        minor_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900] + list(
            np.arange(2e3, t[-1], 1e3))
        cbaxes.yaxis.set_ticks(cbar._locate(minor_ticks), minor=True)

        major_ticks = np.sort(np.hstack((np.asarray([100, 1000]), ts_real)))
        major_ticks_labels = np.sort(np.hstack((np.asarray(major_ticks_labels), ts_real)))

        cbaxes.yaxis.set_ticks(cbar._locate(major_ticks), minor=False)
        cbaxes.set_yticklabels([(f'{num:0.0f}' if num in major_ticks_labels else "") for num in major_ticks])

        for ytick, ytick_label, _t in zip(cbaxes.yaxis.get_major_ticks(), cbaxes.get_yticklabels(), major_ticks):
            if _t in ts_real:
                color = cmap(norm(_t))
                ytick_label.set_color(color)
                ytick_label.set_fontweight('bold')
                ytick.tick2line.set_color(color)
                ytick.tick2line.set_markersize(5)
                # ytick.tick2line.set_markeredgewidth(2)

        if LED_source is not None:
            ax_sec = setup_twin_x_axis(ax, ylim=(None, LED_source.y.max() * 3), y_label="", y_major_locator=FixedLocator([]))
            ax_sec.fill(LED_source.x, LED_source.y, facecolor='gray', alpha=0.5)
            ax_sec.plot(LED_source.x, LED_source.y, color='black', ls='dotted', lw=1)

    plt.tight_layout()
    if filepath:
        plt.savefig(fname=filepath, transparent=transparent, dpi=dpi)

    plt.show()

#
# def plot_kinetic_ax(group_item, n_spectra=50, linscale=1, linthresh=100, cmap='jet_r',
#                   major_ticks_labels=(100, 1000), emph_t=(0, 200, 1000), inset_loc=(0.75, 0.1, 0.03, 0.8),
#                   colorbar_label='Time / s', lw=0.5, alpha=0.5, fig_size=(5, 4), y_label='Absorbance', x_label=WL_LABEL,
#                   x_lim=(230, 600), filepath=None, dpi=500, transparent=True, LED_source_xy=(None, None)):
#
#     t = np.asarray(group_item.get_names(), dtype=np.float64)
#     w = group_item[0].data[:, 0]
#
#     fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
#
#     _set_main_axis(ax1, x_label=x_label, y_label=y_label, xlim=x_lim, x_minor_locator=None, y_minor_locator=None)
#     _ = _setup_wavenumber_axis(ax1)
#
#     cmap = cm.get_cmap(cmap)
#     norm = mpl.colors.SymLogNorm(vmin=t[0], vmax=t[-1], linscale=linscale, linthresh=linthresh, base=10, clip=True)
#
#     tsb_idxs = fi(t, emph_t)
#     ts_real = np.round(t[tsb_idxs])
#
#     x_space = np.linspace(0, 1, n_spectra, endpoint=True, dtype=np.float64)
#
#     t_idx_space = fi(t, norm.inverse(x_space))
#     t_idx_space = np.sort(np.asarray(list(set(t_idx_space).union(set(tsb_idxs)))))
#
#     for i in t_idx_space:
#         x_real = norm(t[i])
#         x_real = 0 if np.ma.is_masked(x_real) else x_real
#         ax1.plot(w, group_item[i].data[:, 1], color=cmap(x_real),
#                  lw=1.5 if i in tsb_idxs else lw,
#                  alpha=1 if i in tsb_idxs else alpha,
#                  zorder=1 if i in tsb_idxs else 0)
#
#     cbaxes = ax1.inset_axes(inset_loc)
#
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, cax=cbaxes, orientation='vertical',
#                         format=mpl.ticker.ScalarFormatter(),
#                         label=colorbar_label)
#
#     cbaxes.invert_yaxis()
#
#     minor_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900] + list(
#         np.arange(2e3, t[-1], 1e3))
#     cbaxes.yaxis.set_ticks(cbar._locate(minor_ticks), minor=True)
#
#     major_ticks = np.sort(np.hstack((np.asarray([100, 1000]), ts_real)))
#     major_ticks_labels = np.sort(np.hstack((np.asarray(major_ticks_labels), ts_real)))
#
#     cbaxes.yaxis.set_ticks(cbar._locate(major_ticks), minor=False)
#     cbaxes.set_yticklabels([(f'{num:0.0f}' if num in major_ticks_labels else "") for num in major_ticks])
#
#     for ytick, ytick_label, _t in zip(cbaxes.yaxis.get_major_ticks(), cbaxes.get_yticklabels(), major_ticks):
#         if _t in ts_real:
#             color = cmap(norm(_t))
#             ytick_label.set_color(color)
#             ytick_label.set_fontweight('bold')
#             ytick.tick2line.set_color(color)
#             ytick.tick2line.set_markersize(5)
#             # ytick.tick2line.set_markeredgewidth(2)
#
#     if LED_source_xy[0] is not None and LED_source_xy[1] is not None:
#         x_LED, y_LED = LED_source_xy
#         ax_sec = setup_twin_x_axis(ax, ylim=(None, y_LED.max() * 3), y_label="", y_major_locator=FixedLocator([]))
#         ax_sec.fill(x_LED, y_LED, facecolor='gray', alpha=0.5)
#         ax_sec.plot(x_LED, y_LED, color='black', ls='dotted', lw=1)
#
#     if filepath:
#         ext = os.path.splitext(filepath)[1].lower()[1:]
#         plt.savefig(fname=filepath, format=ext, transparent=transparent, dpi=dpi)
#
#     plt.show()


def plot_kinetics_no_colorbar(group_item, x_lim=(None, None), y_lim=(None, None), slice_to_plot=slice(0, -1, 5),
                              x_label='Time / s', y_label='$A$', cmap='jet', darkens_factor_cmap=1, colors=None,
                              x_major_locator=None, x_minor_locator=None,
                              y_major_locator=None, y_minor_locator=None,
                              add_wn_axis=True, lw=1.5, ls='-', plot_zero_line=True,
                              label_format_fcn=lambda name: name,
                              legend_loc='best', legend_spacing=0.2, legend_columns=1, legend_column_spacing=2,
                              legend_entry_prefix='pH = ', legend_entry_postfix='', plot_legend_line=True,
                              fig_size=(5.5, 4.5),
                              dpi=500, filepath=None, transparent=True):

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    x = group_item[0].data[:, 0]
    sel_items = group_item[slice_to_plot]

    x_range = (x_lim[0] if x_lim[0] is not None else x[0], x_lim[1] if x_lim[1] is not None else x[-1])

    set_main_axis(ax, x_label=x_label, y_label=y_label, xlim=x_range, ylim=y_lim,
                  x_major_locator=x_major_locator, x_minor_locator=x_minor_locator,
                  y_major_locator=y_major_locator, y_minor_locator=y_minor_locator)

    if add_wn_axis:
        _ = setup_wavenumber_axis(ax, x_major_locator=MultipleLocator(0.5))

    _cmap = cm.get_cmap(cmap, len(sel_items))

    if plot_zero_line:

        ax.axhline(0, x_range[0], x_range[1], ls='--', color='black', lw=1)

    for i, item in enumerate(sel_items):
        if colors is None:
            color = np.asarray(c.to_rgb(_cmap(i))) * darkens_factor_cmap
            color[color > 1] = 1
        else:
            color = colors[i % len(colors)]

        ax.plot(item.x, item.y, color=color, lw=lw, ls=ls,
                label=f'{legend_entry_prefix}{label_format_fcn(item.name)}{legend_entry_postfix}')

    l = ax.legend(loc=legend_loc, frameon=False, labelspacing=legend_spacing, ncol=legend_columns,
                  handlelength=None if plot_legend_line else 0, handletextpad=None if plot_legend_line else 0,
                  columnspacing=legend_column_spacing)

    for i, text in enumerate(l.get_texts()):
        # text.set_ha('right')
        text.set_color(_cmap(i))

    ax.set_axisbelow(False)
    ax.yaxis.set_ticks_position('both')

    plt.tight_layout()

    if filepath:
        ext = os.path.splitext(filepath)[1].lower()[1:]
        plt.savefig(fname=filepath, format=ext, transparent=transparent, dpi=dpi)
    else:
        plt.show()


def plot_EEMs(EEM_group_items: list, n_rows: int = None, n_cols: int = None, log_z: bool = False, transform2wavenumber=True,
              fig_size_one_graph=(5.5, 4), x_lim=(None, None), y_lim=(None, None), z_lim=(1, None),  filepath=None, dpi=500,
              transparent=False, show_title=True, cmap='hot_r', z_label='Counts', x_major_locators=(None, None), x_minor_locators=(None, None),
              y_major_locators=(None, None), y_minor_locators=(None, None)):

    """This will assume that excitation wavelengths are used as names for individual spectra
    x a y lims are in given wavelengths, despite the possible recalculation to wavenumber."""

    n = len(EEM_group_items)  # number of EEMs to plot

    if n_rows is None and n_cols is None:  # estimate the n_rows and n_cols from the sqrt of number of graphs
        sqrt = n ** 0.5
        n_rows = int(np.ceil(sqrt))
        n_cols = int(sqrt)
    elif n_rows is None and n_cols is not None:
        n_rows = int(np.ceil(n / n_cols))
    elif n_rows is not None and n_cols is None:
        n_cols = int(np.ceil(n / n_rows))

    # assert n_rows * n_cols >= n  # not necessary, if the condition is not valid, fewer plots will be plotted

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size_one_graph[0] * n_cols, fig_size_one_graph[1] * n_rows))

    axes = axes.flatten() if np.iterable(axes) else [axes]

    t2w = lambda x: 1e3 / x  # function that transforms wavelength into 10^4 cm-1

    for ax, item in zip(axes, EEM_group_items):

        em_wls, ex_wls, mat = group2mat(item)  # convert group to matrix and extracts ex. wavelengths
        if ex_wls is None:
            raise ValueError(f'Excitation wavelengths of {item.name} could not be extracted from spectra names.')

        x, y = em_wls, ex_wls

        # emission wavelengths limits
        xlim0 = x_lim[0] if x_lim[0] is not None else x[0]
        xlim1 = x_lim[1] if x_lim[1] is not None else x[-1]

        # excitation wavelengths limits
        ylim0 = y_lim[0] if y_lim[0] is not None else y[0]
        ylim1 = y_lim[1] if y_lim[1] is not None else y[-1]

        x_label_down, x_label_top = 'Em. wavelength / nm', 'Em. wavenumber / $10^4$ cm$^{-1}$'
        y_label_left, y_label_right = 'Ex. wavelength / nm', 'Ex. wavenumber / $10^4$ cm$^{-1}$'

        if transform2wavenumber:
            x, y = t2w(x), t2w(y)
            xlim0, xlim1 = t2w(xlim0), t2w(xlim1)
            ylim0, ylim1 = t2w(ylim0), t2w(ylim1)

            # switch the labels
            x_label_down, x_label_top = x_label_top, x_label_down
            y_label_left, y_label_right = y_label_right, y_label_left

        _set_main_axis(ax, xlim=(xlim0, xlim1), ylim=(ylim0, ylim1),
                       y_label=y_label_left,
                       x_label=x_label_down, direction='out',
                       x_major_locator=x_major_locators[0],
                       y_major_locator=y_major_locators[0],
                       x_minor_locator=x_minor_locators[0],
                       y_minor_locator=y_minor_locators[0])

        if log_z:  # use log of z axis
            # mat[mat < 0] = 0
            zmin = mat.max() * 1e-3 if z_lim[0] is None else z_lim[0]  # 3 orders lower than max as default value
        else:
            zmin = mat.min() if z_lim[0] is None else z_lim[0]  # for linear plot, min as default value

        zmax = mat.max() if z_lim[1] is None else z_lim[1]

        # add left axis
        lambda_ax = ax.secondary_xaxis('top', functions=(t2w, t2w))
        lambda_ax.tick_params(which='both', direction='out', zorder=1000)
        if x_major_locators[1] is not None:
            lambda_ax.xaxis.set_major_locator(x_major_locators[1])  # FixedLocator([500, 600, ...])
        if x_minor_locators[1] is not None:
            lambda_ax.xaxis.set_minor_locator(x_minor_locators[1])
        lambda_ax.set_xlabel(x_label_top)

        # add right axis
        lambda_ax2 = ax.secondary_yaxis('right', functions=(t2w, t2w))
        lambda_ax2.tick_params(which='both', direction='out', zorder=1000)
        if y_major_locators[1] is not None:
            lambda_ax2.yaxis.set_major_locator(y_major_locators[1])  # MultipleLocator(20)
        if y_minor_locators[1] is not None:
            lambda_ax2.yaxis.set_minor_locator(y_minor_locators[1])  # AutoMinorLocator(2)
        lambda_ax2.set_ylabel(y_label_right)

        # norm for z values
        norm = mpl.colors.LogNorm(vmin=zmin, vmax=zmax, clip=True) if log_z else mpl.colors.Normalize(vmin=zmin,
                                                                                                       vmax=zmax,
                                                                                                       clip=True)
        _x, _y = np.meshgrid(x, y)
        mappable = ax.pcolormesh(_x, _y, mat.T, norm=norm, cmap=cmap, shading='auto')
        fig.colorbar(mappable, ax=ax, label=z_label, pad=0.17, format=None if log_z else '%.0e')

        if show_title:
            ax.set_title(item.name)

        # if x_major_formatter:
        #     ax_data.xaxis.set_major_formatter(x_major_formatter)
        #     ax_res.xaxis.set_major_formatter(x_major_formatter)

    plt.tight_layout()

    if filepath:
        ext = os.path.splitext(filepath)[1].lower()[1:]
        plt.savefig(fname=filepath, format=ext, transparent=transparent, dpi=dpi)

    plt.show()


def plot_fit(data_item, fit_item, residuals_item, symlog=False, linscale=1, linthresh=100,
              lw_data=0.5, lw_fit=1.5, fig_size_one_graph=(5, 4), y_label='$\\Delta$A', x_label='Time / $\\mu$s',
              x_lim=(None, None), t_mul_factor=1, y_lim=(None, None), x_margin=1, y_margin=1.05, filepath=None, dpi=500,
              transparent=False, x_major_formatter=ScalarFormatter(), x_major_locator=None, y_major_locator=None,
              data_color='red', show_title=True):

    plot_fits([data_item], [fit_item], [residuals_item], n_rows=1, n_cols=1, symlog=symlog, linscale=linscale,
              linthresh=linthresh, lw_data=lw_data, lw_fit=lw_fit, fig_size_one_graph=fig_size_one_graph,
              y_label=y_label, x_label=x_label, x_lim=x_lim, t_mul_factor=t_mul_factor, y_lim=y_lim, x_margin=x_margin,
              y_margin=y_margin, filepath=filepath, dpi=dpi, transparent=transparent,
              x_major_formatter=x_major_formatter, x_major_locator=x_major_locator, y_major_locator=y_major_locator,
              data_color=data_color, show_title=show_title)


def plot_fits(data_group, fit_group, residuals_group, n_rows=None, n_cols=None, symlog=False, linscale=1, linthresh=100,
              lw_data=0.5, lw_fit=1.5, fig_size_one_graph=(5, 4), y_label='$\\Delta$A', x_label='Time / $\\mu$s',
              x_lim=(None, None), t_mul_factor=1, y_lim=(None, None), x_margin=1, y_margin=1.05, filepath=None, dpi=500,
              transparent=False, x_major_formatter=ScalarFormatter(), x_major_locator=None, y_major_locator=None,
              data_color='red', show_title=True):

    n = len(data_group)
    assert n == len(fit_group) == len(residuals_group)

    if n_rows is None and n_cols is None:  # estimate the n_rows and n_cols from the sqrt of number of graphs
        sqrt = n ** 0.5
        n_rows = int(np.ceil(sqrt))
        n_cols = int(sqrt)
    elif n_rows is None and n_cols is not None:
        n_rows = int(np.ceil(n / n_cols))
    elif n_rows is not None and n_cols is None:
        n_cols = int(np.ceil(n / n_rows))

    # assert n_rows * n_cols >= n  # not necessary, if the condition is not valid, fewer plots will be plotted

    fig = plt.figure(figsize=(fig_size_one_graph[0] * n_cols, fig_size_one_graph[1] * n_rows))

    outer_grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.25, hspace=0.3)

    for og, data, fit, res in zip(outer_grid, data_group, fit_group, residuals_group):

        # nice tutorial about gridspec here https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.gridspec.GridSpecFromSubplotSpec.html
        # each unit consist of two graphs - data and residuals
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=og, wspace=0.1, hspace=0.1,
                                                      height_ratios=(4, 1))

        ax_data = fig.add_subplot(inner_grid[0])
        ax_res = fig.add_subplot(inner_grid[1])

        t_data = data.data[:, 0] * t_mul_factor

        _x_lim = list(x_lim)
        _y_lim = list(y_lim)

        _x_lim[0] = data.data[0, 0] * x_margin * t_mul_factor if _x_lim[0] is None else _x_lim[0]
        _x_lim[1] = data.data[-1, 0] * x_margin * t_mul_factor if _x_lim[1] is None else _x_lim[1]

        _y_lim[0] = data.data[:, 1].min() * y_margin if _y_lim[0] is None else _y_lim[0]
        _y_lim[1] = data.data[:, 1].max() * y_margin if _y_lim[1] is None else _y_lim[1]

        _set_main_axis(ax_data, x_label="", y_label=y_label, xlim=_x_lim, ylim=_y_lim, x_major_locator=x_major_locator,
                       y_major_locator=y_major_locator)
        _set_main_axis(ax_res, x_label=x_label, y_label='res.', xlim=_x_lim, x_minor_locator=None, y_minor_locator=None)

        # plot zero lines
        ax_data.axline((0, 0), slope=0, ls='--', color='black', lw=0.5)
        ax_res.axline((0, 0), slope=0, ls='--', color='black', lw=0.5)

        ax_data.tick_params(labelbottom=False)

        if show_title:
            ax_data.set_title(data.name)
        ax_data.plot(t_data, data.data[:, 1], lw=lw_data, color=data_color)
        ax_data.plot(fit.data[:, 0] * t_mul_factor, fit.data[:, 1], lw=lw_fit, color='black')
        ax_res.plot(res.data[:, 0] * t_mul_factor, res.data[:, 1], lw=lw_data, color=data_color)

        ax_data.set_axisbelow(False)
        ax_res.set_axisbelow(False)

        ax_data.yaxis.set_ticks_position('both')
        ax_data.xaxis.set_ticks_position('both')

        ax_res.yaxis.set_ticks_position('both')
        ax_res.xaxis.set_ticks_position('both')

        if symlog:
            ax_data.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)
            ax_res.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linscale=linscale, linthresh=linthresh)
            ax_data.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
            ax_res.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))

        if x_major_formatter:
            ax_data.xaxis.set_major_formatter(x_major_formatter)
            ax_res.xaxis.set_major_formatter(x_major_formatter)

    if filepath:
        ext = os.path.splitext(filepath)[1].lower()[1:]
        plt.savefig(fname=filepath, format=ext, transparent=transparent, dpi=dpi)

    plt.show()


def save_group(group_item, fname='', delimiter='\t', encoding='utf8'):
    """Data will be saved x-axis explicit"""

    x, y, mat = group2mat(group_item)

    mat = np.vstack((x, mat.T))
    buffer = delimiter + delimiter.join(f"{num}" for num in y) + '\n'
    buffer += '\n'.join(delimiter.join(f"{num}" for num in row) for row in mat.T)

    with open(fname, 'w', encoding=encoding) as f:
        f.write(buffer)


def reaction_QY_relative(sample_items, actinometer_items, QY_act=1, irradiation_spectrum=None,
                         irradiation_wavelength=None, integration_range=(None, None), V_solution=1,
                         conc_calc_range=(None, None), samples_times=None, actinometers_times=None,
                         c_0_samples=1, c_0_acts=1, integration_method='trapz'):
    """
    TODO.....

    Initial spectrua of both sample and actinometers must be at time = 0 (before irradiaiton).

    if conc_calc_range is (None, None), all spectrum will be taken for calclation of conccentration
    V solution

    integration method works for spectra integration as well as for cumulative time integration


    :param sample_items:
    :param actinometer_items:
    :param irradiation_spectrum:
    :param irradiation_wavelength:
    :param integration_range:
    :param sample_times:
    :param actinometer_times:
    :param integration_method: trapz or simps - simpsons rule
    :return:
    """

    # type checking
    if sample_items is None or actinometer_items is None:
        raise ValueError("Arguments sample_items and actinometer_items must not be None.")

    if not isinstance(sample_items, (list, tuple, SpectrumList)) and \
            not isinstance(actinometer_items, (list, tuple, SpectrumList)):
        raise ValueError("Arguments sample_items and actinometer_items must be type list, tuple or SpectrumList.")

    if isinstance(sample_items, (list, tuple)) and not isinstance(sample_items[0], SpectrumList):
        raise ValueError("Entries of sample_items must be type of SpectrumList.")

    if isinstance(actinometer_items, (list, tuple)) and not isinstance(actinometer_items[0], SpectrumList):
        raise ValueError("Entries of actinometer_items must be type of SpectrumList.")

    if irradiation_wavelength is None and irradiation_spectrum is None:
        raise ValueError("Argument irradiation_spectrum or irradiation_wavelength must be provided.")

    if irradiation_spectrum is not None and irradiation_wavelength is not None:
        raise ValueError("Only one argument of irradiation_spectrum or irradiation_wavelength must be provided.")

    if irradiation_spectrum is None and not isinstance(irradiation_wavelength, (int, float)):
        raise ValueError("Argument irradiation_wavelength must be type of int or float.")

    if irradiation_wavelength is None and not isinstance(irradiation_spectrum, (Spectrum, np.ndarray, list)):
        raise ValueError("Argument irradiation_spectrum must be type of Spectrum, ndarray or list.")

    if not isinstance(integration_range, tuple) or len(integration_range) != 2:
        raise ValueError("Argument integration_range must be type of tuple and have length of 2.")

    samples = [sample_items] if isinstance(sample_items, SpectrumList) else sample_items
    acts = [actinometer_items] if isinstance(actinometer_items, SpectrumList) else actinometer_items

    if irradiation_spectrum:
        irr_sp_x = None
        if isinstance(irradiation_spectrum, Spectrum):
            irr_sp_x = irradiation_spectrum.data[:, 0]
            irr_sp_y = irradiation_spectrum.data[:, 1]
        else:
            irr_sp_y = np.asarray(irradiation_spectrum)

        x0, x1 = integration_range
        if x0 is not None and x1 is not None and x0 > x1:
            x0, x1 = x1, x0

        def abs_photons(data):
            start = 0
            end = data.shape[0]
            start_sp = 0
            end_sp = irr_sp_y.shape[0]
            if x0 is not None:
                start = fi(data[:, 0], x0)
                start_sp = fi(irr_sp_y, x0)
            if x1 is not None:
                end = fi(data[:, 0], x1) + 1
                end_sp = fi(irr_sp_y, x1) + 1

            if start - end != start_sp - end_sp:
                if irr_sp_x is None:
                    raise ValueError("Irradiation spectrum and data does not have equal dimension.")
                irr = np.interp(data[:, 0], irr_sp_x, irr_sp_y)  # interpolate to match data if x vals are provided
            else:
                irr = irr_sp_y[start_sp:end_sp]  # slice the irradiation spectrum to match the data

            x = data[start:end, 0]
            y = data[start:end, 1]

            abs_light = (1 - 10 ** -y) * irr  # (1 - 10 ** -A) * I_irr

            if integration_method == 'trapz':
                return np.trapz(abs_light, x)  # integrate using trapezoidal rule
            else:
                return simps(abs_light, x)  # integrate using simpsons rule

    else:  # only irradiation wavelength
        def abs_photons(data):
            idx = fi(data[:, 0], irradiation_wavelength)
            return 1 - 10 ** -data[idx, 1]  # 1 - 10 ** -A

    _sample_times = [] if samples_times is None else samples_times
    _acs_times = [] if actinometers_times is None else actinometers_times

    if samples_times is None:
        for sample in samples:
            _sample_times = np.asarray([float(sp.name) for sp in sample])

    if actinometers_times is None:
        for act in acts:
            _acs_times = np.asarray([float(sp.name) for sp in act])

    _c_0_samples = [c_0_samples] * len(samples) if isinstance(c_0_samples, (int, float)) else c_0_samples
    _c_0_acts = [c_0_acts] * len(acts) if isinstance(c_0_acts, (int, float)) else c_0_acts

    def cumintegrate(y, x, initial=0):
        if integration_method == 'trapz':
            return cumtrapz(y, x, initial=initial)
        else:
            # simpsons rule
            raise NotImplementedError()  # TODO----->

    def calc_c(unknown_sp_data, sp_data_c0, c0=1):
        """
        Calculation of concentariton by least squares.

        :param unknown_sp_data:
        :param sp_data_c0:
        :param c0:
        :return:
        """
        assert unknown_sp_data.shape == sp_data_c0.shape
        x0, x1 = conc_calc_range
        start = 0
        end = sp_data_c0.shape[0]
        if x0 is not None:
            start = fi(sp_data_c0[:, 0], x0)
        if x1 is not None:
            end = fi(sp_data_c0[:, 0], x1) + 1

        # for min || xA - B ||_2^2 for scalar x, x = sum(A*B) / sum(A*A)
        a = sp_data_c0[start:end, 1]
        b = unknown_sp_data[start:end, 1]

        return c0 * (a * b).sum() / (a * a).sum()

    results_sample = SpectrumList(name='Results of samples')
    results_act = SpectrumList(name='Results of actinometers')

    def calculate_line(sl, times, c0):
        # calculate the amount of absorbed photons
        qs = np.asarray([abs_photons(sp.data) for sp in sl])
        # calculate the time-dependent concentration
        c0s = np.asarray([calc_c(sp.data, sl[0].data, c0) for sp in sl])
        # Delta n = (c(t=0) - c(t)) * V
        d_n_dec = (c0s[0] - c0s) * V_solution
        # cumulative integration of light absorbed: int_0^t q(t') dt'
        np_abs = cumintegrate(qs, times, initial=0)

        return Spectrum.from_xy_values(np_abs, d_n_dec, name=f'Result for {sl.name}')

    for sample, s_times, c0_sample in zip(samples, _sample_times, _c_0_samples):
        results_sample.children.append(calculate_line(sample, s_times, c0_sample))

    for act, act_times, c0_act in zip(acts, _acs_times, _c_0_acts):
        results_act.children.append(calculate_line(act, act_times, c0_act))

    QYs = []

    # calculate for each combination of sample and actinometer kinetics
    for res_sample in results_sample:
        slope_sam, intercept_sam, r_sam, _, err_sam = linregress(res_sample.x, res_sample.y)

        for res_act in results_act:
            slope_act, intercept_act, r_act, _, err_act = linregress(res_act.x, res_act.y)

            # use uncertainty package to automatically propagate errors
            # QY is type of ufloat - uncertainties.core.Variable
            QY = QY_act * ufloat(slope_sam, err_sam) / ufloat(slope_act, err_act)

            QYs.append(QY)

    average_QY = sum(QYs) / len(QYs)

    add_to_list(results_sample)
    add_to_list(results_act)






def bcorr_1D(item, first_der_tresh=1e-4, second_der_tresh=0.1):
    """Gradient based baseline correction"""

    x = item.data[:, 0].copy()
    y = item.data[:, 1].copy()

    grad1 = np.gradient(y, x)  # first central derivative
    grad2 = np.gradient(grad1, x)  # second central derivative

    grad1, grad2 = grad1 / grad1.max(), grad2 / grad2.max()

    zero_idxs = np.argwhere(
        (grad1 < first_der_tresh) & (grad1 > -first_der_tresh) &
        (grad2 < second_der_tresh) & (grad2 >= 0)
    )

    zero_idxs = zero_idxs.squeeze()

    baseline = np.interp(x, x[zero_idxs], y[zero_idxs])

    sp = Spectrum.from_xy_values(x, baseline, f'{item.name} baseline')
    UserNamespace.instance.add_items_to_list(sp)


class UserNamespace:
    instance = None

    def __init__(self, main):

        self.main = main
        UserNamespace.instance = self
        self.tw = self.main.tree_widget

        # execute first commands
        self.main.console.execute_command(
            """
            import numpy as np
            from spectramanipulator.user_namespace import *
            # from spectramanipulator.spectrum import fi, group2mat
            import matplotlib.pyplot as plt
            %matplotlib inline
            
            # setup important methods
            add_to_list = UserNamespace.instance.tw.add_to_list
            load_kinetic = UserNamespace.instance.tw.load_kinetic
            load_kinetics = UserNamespace.instance.tw.load_kinetics
            import_files = UserNamespace.instance.tw.import_files
            
            """
        )

        # from IPython.display import display, Math, Latex\n

        self.main.console.push_variables(
            {
                'main': self.main,
                'tree_widget': self.main.tree_widget,
                'item': self.main.tree_widget.myModel.root
            }
        )

    # def add_items_to_list(self, spectra):
    #     """
    #     Copies all spectra and import them to the treewidget
    #     :param spectra: input parameter can be single spectrum object, or hierarchic list of spectra
    #     """
    #
    #     # self.main.tree_widget.get
    #
    #     if spectra.__class__ == Spectrum:
    #         self.main.tree_widget.import_spectra([spectra])
    #         return
    #
    #     if spectra.__class__.__name__ == 'SpectrumItem':
    #         self.main.tree_widget.import_spectra([spectra.__copy__()])
    #         return
    #
    #     if isinstance(spectra, list):
    #         self.main.tree_widget.import_spectra(spectra)
    #         return
    #
    #     if spectra.__class__.__name__ == 'SpectrumItemGroup' or spectra.__class__.__name__ == 'SpectrumList':
    #         l = []
    #         for sp in spectra:
    #             new_sp = sp.__copy__()
    #             new_sp.group_name = spectra.name
    #             l.append(new_sp)
    #
    #         self.main.tree_widget.import_spectra([l])
    #         return
