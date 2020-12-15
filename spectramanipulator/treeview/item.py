from PyQt5.QtCore import Qt
from spectramanipulator.spectrum import Spectrum, SpectrumList
import spectramanipulator.user_namespace as un

import numpy as np

from copy import deepcopy


class GenericItem:

    def __init__(self, name, info, parent=None):

        self.check_state = Qt.Unchecked

        self.name = str(name)
        self.info = str(info)

        self.parent = parent
        self.setParent(parent)

        self.children = []

    def is_root(self):
        return self.parent is None

    def isChecked(self):
        return self.check_state == Qt.Checked or self.check_state == Qt.PartiallyChecked

    def setParent(self, parent, row=None):
        if parent is not None:
            self.parent = parent
            self.parent.appendChild(self, row)

    def appendChild(self, child, row=None):
        self.children.insert(row if row is not None else len(self.children), child)
        # self.children.append(child)

    def childAtRow(self, row):
        try:
            return self.children[row]
        except IndexError:
            # print("index error")
            return

    def move_child(self, row_of_child: int, to_row=0):
        self.children.insert(to_row, self.children.pop(row_of_child))

    def rowOfChild(self, child):
        for i, item in enumerate(self.children):
            if item == child:
                return i
        return -1

    def removeChildAtRow(self, row):
        # try:

        del self.children[row]

        # value = self.children[row]
        # self.children.remove(value)
        # except:
        #     pass

        return True

    def row(self):
        if self.parent is not None:
            return self.parent.rowOfChild(self)

    def removeChild(self, child):
        self.children.remove(child)

    def root(self):
        parent = self.parent
        if parent is None:
            return self
        while parent.parent is not None:
            parent = parent.parent
        return parent

    def __len__(self):
        return len(self.children)

    def __getitem__(self, item):
        return self.children[item]

    def __iter__(self):
        return iter(self.children)

    def add_to_list(self, spectra=None):
        if self.__class__ == SpectrumItem or self.__class__ == SpectrumItemGroup:
            un.add_to_list(self if spectra is None else spectra)
            # print('add_to_list - generic')

    def _redraw_all_spectra(self):
        un.redraw_all_spectra()

        # print('_redraw_all_spectra - generic')

    def _update_view(self):
        un.update_view()
        # print('_update_view - generic')


class SpectrumItem(GenericItem, Spectrum):

    def __init__(self, name, info='', parent=None, color=None, line_width=None, line_alpha=255,
                 line_type=None, symbol=None, symbol_brush=None, sym_brush_alpha=255, symbol_fill=None,
                 sym_fill_alpha=255, symbol_size=8, plot_legend=True):

        super(SpectrumItem, self).__init__(name, info, parent)

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
    def from_spectrum(cls, spectrum, info='', parent=None):
        si = cls(spectrum.name, info, parent,  # for backward compatibility
                 color=getattr(spectrum, 'color', None),
                 line_width=getattr(spectrum, 'line_width', None),
                 line_alpha=getattr(spectrum, 'line_alpha', 255),
                 line_type=getattr(spectrum, 'line_type', None),
                 symbol=getattr(spectrum, 'symbol', None),
                 symbol_brush=getattr(spectrum, 'symbol_brush', None),
                 sym_brush_alpha=getattr(spectrum, 'sym_brush_alpha', 255),
                 symbol_fill=getattr(spectrum, 'symbol_fill', None),
                 sym_fill_alpha=getattr(spectrum, 'sym_fill_alpha', 255),
                 symbol_size=getattr(spectrum, 'symbol_size', 8),
                 plot_legend=getattr(spectrum, 'plot_legend', True))

        si.filepath = spectrum.filepath
        si.data = spectrum.data
        # del spectrum

        return si

    @classmethod
    def from_xy_values(cls, x_values, y_values, name='', info='', parent=None, color=None, line_width=None, line_alpha=255,
                 line_type=None, symbol=None, symbol_brush=None, sym_brush_alpha=255, symbol_fill=None,
                 sym_fill_alpha=255, symbol_size=8, plot_legend=True):

        sp = cls(name, info, parent, color, line_width, line_alpha, line_type, symbol, symbol_brush, sym_brush_alpha,
                 symbol_fill, sym_fill_alpha, symbol_size, plot_legend)

        sp_spectrum = Spectrum.from_xy_values(x_values, y_values, name='')

        sp.data = sp_spectrum.data
        del sp_spectrum

        return sp

    @property
    def x(self):
        return self.data[:, 0]

    @x.setter
    def x(self, array):
        self.data[:, 0] = array
        self._redraw_all_spectra()
        self._update_view()

    @property
    def y(self):
        return self.data[:, 1]

    @y.setter
    def y(self, array):
        self.data[:, 1] = array
        self._redraw_all_spectra()
        self._update_view()

    def spacing(self):
        """
        Returns a type and spacing of the x values:
            * r - probably regular spacing for all points
            * i - probably irregular spacing for all points

        The r/i distinction is made by checking differences between first and the end x values.
        Value is computed as average spacing:  x_max - x_min / number of points

        :return: str
        """

        spacing = (self.data[-1, 0] - self.data[0, 0]) / (self.data.shape[0] - 1)  # average spacing
        s_type = 'r'  # regular spacing
        if self.data.shape[0] > 2:
            # irregular spacing
            x_diff = self.data[1:, 0] - self.data[:-1, 0]  # x differences
            if not np.allclose(x_diff, x_diff[0]):
                s_type = 'i'

        return s_type + "{:.3g}".format(spacing)

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

    def set_default_style(self):
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

        self._redraw_all_spectra()

    def is_in_group(self):
        return not self.parent.is_root()

    def is_top_level(self):
        return self.parent.is_root()

    def __copy__(self):
        return deepcopy(self)


class SpectrumItemGroup(GenericItem, SpectrumList):

    def __init__(self, name='', info='', parent=None):
        super(SpectrumItemGroup, self).__init__(name, info, parent)

        # self.setup_fcn()

    def set_names(self, names):
        # super(SpectrumItemGroup, self).set_names(names)
        for sp, new_name in zip(self, names):
            sp.name = str(new_name)
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

        for sp in self:
            sp.set_plot_legend(plot_legend, False)

    def set_style(self, color=None, line_width=None, line_type=None):
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

        for sp in self:
            sp.set_style(color, line_width, line_type, False)

        self._redraw_all_spectra()

    def is_top_level(self):
        return True
