

from PyQt5 import QtGui
import pyqtgraph as pg

from spectramanipulator.pyqtgraphmodif.legend_item import LegendItem
from spectramanipulator.pyqtgraphmodif.plot_item import PlotItem
from spectramanipulator.pyqtgraphmodif.view_box import ViewBox

from pyqtgraph.exporters import ImageExporter
from pyqtgraph.exporters import SVGExporter

from spectramanipulator.settings import Settings

from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QColor


# subclassing of PlotWidget
class PlotWidget(pg.PlotWidget):
    _instance = None

    def __init__(self, parent=None, coordinates_func=None):

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.plotted_items = {}  # dictionary with keys as spectrum objects and values as plot data items
        self.plotted_fits = {}  #

        self.plotItem = PlotItem(viewBox=ViewBox(self.plotted_items))

        # use our modified plotItem
        super(PlotWidget, self).__init__(parent, plotItem=self.plotItem)

        PlotWidget._instance = self

        self.coordinates_func = coordinates_func

        self.legend = None
        # self.add_legend(spacing=Settings.legend_spacing)

        self.img_exporter = ImageExporter(self.plotItem)
        self.svg_exporter = SVGExporter(self.plotItem)

        self.lr_item = None  # linear region item

        self.update_settings()
        self.plotItem.setDownsampling(ds=True, auto=True, mode='subsample')
        self.plotItem.setClipToView(True)

    def update_settings(self):

        self.clear_plots()
        self.add_legend(spacing=Settings.legend_spacing)

        pg.setConfigOptions(antialias=Settings.antialiasing)
        self.plotItem.showAxis('top', show=True)
        self.plotItem.showAxis('right', show=True)

        self.plotItem.setTitle(Settings.graph_title, size="{}pt".format(Settings.graph_title_font_size))

        self.plotItem.setLabel('left', text=Settings.left_axis_label, units=Settings.left_axis_unit)
        self.plotItem.setLabel('bottom', text=Settings.bottom_axis_label, units=Settings.bottom_axis_unit)
        self.plotItem.showGrid(x=Settings.show_grid, y=Settings.show_grid, alpha=Settings.grid_alpha)

        left_label_font = QtGui.QFont()
        left_label_font.setPixelSize(Settings.left_axis_font_size)

        self.plotItem.getAxis('left').label.setFont(left_label_font)

        bottom_label_font = QtGui.QFont()
        bottom_label_font.setPixelSize(Settings.bottom_axis_font_size)

        self.plotItem.getAxis('bottom').label.setFont(bottom_label_font)

    def plot(self, item, **kwargs):
        """kwargs are passed to plotItem.plot function"""

        # if spectrum is already plotted, update the data and style
        if item in self.plotted_items:
            self.plotted_items[item].setData(item.data, **kwargs)
            if 'zValue' in kwargs:
                self.plotted_items[item].setZValue(kwargs['zValue'])
            return

        pi = self.plotItem.plot(item.data, **kwargs)
        self.plotted_items[item] = pi

    def update_items_data(self, items: list):
        """Updates the plots of items."""
        for item in items:
            try:
                self.plotted_items[item].setData(item.data)
            except KeyError:
                continue

    def remove(self, spectrum):
        if spectrum in self.plotted_items:
            self.removeItem(self.plotted_items[spectrum])  # remove from pyqtgraph
            del self.plotted_items[spectrum]  # remove entry in dictionary

    def clear_plots(self):
        """Removes all plots except of linear region item and fits"""
        # self.plotted_spectra.clear()
        for item in self.plotted_items.values():
            self.removeItem(item)

        self.plotted_items.clear()

    @classmethod
    def plotted_items(cls):
        if cls._instance is not None:
            return cls._instance.plotted_items

    @classmethod
    def plot_fit(cls, item, **kwargs):
        self = cls._instance
        if self is None:
            return

        if item in self.plotted_fits:
            self.plotted_fits[item].setData(item.data, **kwargs)
            if 'zValue' in kwargs:
                self.plotted_fits[item].setZValue(kwargs['zValue'])
            return

        pi = self.plotItem.plot(item.data, **kwargs)
        self.plotted_fits[item] = pi

    @classmethod
    def remove_fits(cls, items: list):
        self = cls._instance
        if self is None:
            return

        for item in items:
            if item in self.plotted_fits:
                self.removeItem(self.plotted_fits[item])
                del self.plotted_fits[item]

    @classmethod
    def remove_all_fits(cls):
        self = cls._instance
        if self is None:
            return

        for val in self.plotted_fits.values():
            self.removeItem(val)

        self.plotted_fits.clear()

    @classmethod
    def add_linear_region(cls, region=None, bounds=None, brush=None, orientation='vertical', z_value=-10):
        """bounds is tuple or None"""

        self = cls._instance
        if self is None:
            return

        if region is None:
            region = self.getViewBox().viewRange()[0]
            f = 0.87
            diff = region[1] - region[0]
            region[0] += (1 - f) * diff
            region[1] += f * diff

        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0, 20)) if brush is None else brush

        if self.lr_item is None:
            self.lr_item = pg.LinearRegionItem(region, orientation=orientation, brush=brush, bounds=bounds)
            self.addItem(self.lr_item)

        self.lr_item.setRegion(region)
        if bounds is not None:
            self.lr_item.setBounds(bounds)
        self.lr_item.setBrush(brush)
        self.lr_item.setZValue(z_value)

        return self.lr_item

    @classmethod
    def remove_linear_region(cls):
        self = cls._instance
        if self is None:
            return

        if self.lr_item is not None:
            self.removeItem(self.lr_item)
            self.lr_item = None

    @classmethod
    def set_lr_x_range(cls, x0, x1):
        self = cls._instance
        if self is None:
            return

        if self.lr_item is not None:
            self.lr_item.setRegion((x0, x1))

    @classmethod
    def get_view_range(cls):
        if cls._instance is None:
            return

        x0, x1 = cls._instance.getViewBox().viewRange()[0]
        y0, y1 = cls._instance.viewRange()[1]

        return x0, x1, y0, y1

    def save_plot_to_clipboard_as_png(self):
        self.img_exporter.export(copy=True)

    def save_plot_to_clipboard_as_svg(self):
        self.svg_exporter.export(copy=True)

    def leaveEvent(self, ev):
        super(PlotWidget, self).leaveEvent(ev)
        self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, ev):
        super(PlotWidget, self).mouseMoveEvent(ev)

        pos = ev.pos()

        in_scene = self.plotItem.sceneBoundingRect().contains(pos)
        in_legend = self.legend.sceneBoundingRect().contains(pos)

        in_lin_reg = False
        in_lin_reg_line = False
        lin_reg_moving = False
        if self.lr_item is not None:
            in_lin_reg = self.lr_item.sceneBoundingRect().contains(pos)
            line1, line2 = self.lr_item.lines
            in_lin_reg_line = line1.sceneBoundingRect().contains(pos) or line2.sceneBoundingRect().contains(pos)
            lin_reg_moving = self.lr_item.moving or line1.moving or line2.moving

        if in_scene:
            try:
                mouse_point = self.plotItem.vb.mapSceneToView(pos)
                n = Settings.coordinates_sig_figures
                # double format with n being the number of significant figures of a number
                self.coordinates_func(f"x={{:.{n}g}}, y={{:.{n}g}}".format(mouse_point.x(), mouse_point.y()))
            except:
                pass

        # set the corresponding cursor
        if in_scene and not lin_reg_moving:
            if in_lin_reg_line:
                self.viewport().setCursor(Qt.SizeHorCursor)
            elif in_legend or in_lin_reg:
                self.viewport().setCursor(Qt.SizeAllCursor)
            else:
                self.viewport().setCursor(Qt.CrossCursor)

    def add_legend(self, size=None, spacing=5, offset=(-30, 30)):
        if self.legend is not None:
            self.legend.clear()
            self.plotItem.legend = None
            del self.legend

        self.legend = LegendItem(size, verSpacing=spacing, offset=offset)
        self.legend.setParentItem(self.plotItem.vb)
        self.plotItem.legend = self.legend


