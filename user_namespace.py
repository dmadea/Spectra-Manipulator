from spectrum import Spectrum, SpectrumList
# from copy import deepcopy
from PyQt5.QtWidgets import QApplication
from scipy.linalg import lstsq

# for backward compatibility of smpj files
ItemList = list

import numpy as np
from typing import Iterable
from settings import Settings
import os






def add_to_list(spectra):
    """
    Copies all spectra and imports them to the Tree Widget.

    Parameters
    ----------
    spectra : {:class:`Spectrum`, :class:`SpectrumList`, list, list of lists, SpectrumItemGroup, SpectrumItem}
        The input spectra to be added into Tree Widget.
    """

    if UserNamespace.instance is not None:
        UserNamespace.instance.add_items_to_list(spectra)


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

def _get_D(group):
    D = group[0].data[:, 1]
    for i in range(1, len(group)):
        D = np.vstack((D, group[i].data[:, 1]))
    return D


def calc_Eps(group):
    C = _get_C(group)
    D = _get_D(group)
    wls = group[0].data[:, 0]

    # C needs to be changed to column vector
    ST = lstsq(C.reshape(-1, 1), D)[0]

    # add a spectrum to list
    Spectrum.from_xy_values(wls, ST.flatten(), name=group.name + '-epsilon').add_to_list()




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


def load_kinetics(dir_name, spectra_dir_name='spectra', times_fname='times.txt', blank_spectrum='blank.dx'):
    """given a directory name, it loads all spectra in dir named "spectra",
    if blank is given, it will be subtracted from all spectra, times.txt will contain
    times for all spectra"""

    if UserNamespace.instance is None:
        return

    if not os.path.isdir(dir_name):
        raise ValueError(f'{dir_name}  does not exist!')
    
    spectra_path = os.path.join(dir_name, spectra_dir_name)
    
    if not os.path.isdir(spectra_path):
        raise ValueError(f'{spectra_dir_name}  does not exist in {dir_name}!')

    spectras = []

    for filename in os.listdir(spectra_path):
        spectras.append(os.path.join(spectra_path, filename))

    UserNamespace.instance.main.tree_widget.import_files(spectras)

    times_fname = os.path.join(dir_name, times_fname)
    if os.path.isfile(times_fname):
        UserNamespace.instance.main.tree_widget.import_files(times_fname)


class UserNamespace:
    instance = None

    def __init__(self, main):

        self.main = main
        UserNamespace.instance = self

        # execute first commands
        self.main.console.execute_command(
            "import numpy as np\nfrom user_namespace import *\n"
            "import matplotlib.pyplot as plt\n%matplotlib inline")

        # from IPython.display import display, Math, Latex\n

        self.main.console.push_variables(
            {
                'main': self.main,
                'tree_widget': self.main.tree_widget,
                'item': self.main.tree_widget.myModel.root
            }
        )

    def add_items_to_list(self, spectra):
        """
        Copies all spectra and import them to the treewidget
        :param spectra: input parameter can be single spectrum object, or hierarchic list of spectra
        """

        # self.main.tree_widget.get

        if spectra.__class__ == Spectrum:
            self.main.tree_widget.import_spectra([spectra])
            return

        if spectra.__class__.__name__ == 'SpectrumItem':
            self.main.tree_widget.import_spectra([spectra.__copy__()])
            return

        if isinstance(spectra, list):
            self.main.tree_widget.import_spectra(spectra)
            return

        if spectra.__class__.__name__ == 'SpectrumItemGroup' or spectra.__class__.__name__ == 'SpectrumList':
            l = []
            for sp in spectra:
                new_sp = sp.__copy__()
                new_sp.group_name = spectra.name
                l.append(new_sp)

            self.main.tree_widget.import_spectra([l])
            return
