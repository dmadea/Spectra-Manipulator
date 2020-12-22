

from PyQt5 import QtGui
import pyqtgraph as pg

from spectramanipulator.pyqtgraphmodif.legend_item_modif import LegendItemModif
from spectramanipulator.pyqtgraphmodif.plot_item_modif import PlotItemModif
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

        self.plotted_spectra = []
        self.plotted_items = []

        self.plotItem = PlotItemModif(viewBox=ViewBox(self.plotted_spectra))

        # use our modified plotItem
        super(PlotWidget, self).__init__(parent, plotItem=self.plotItem)

        PlotWidget._instance = self

        self.coordinates_func = coordinates_func

        self.legend = None
        # self.add_legend(spacing=Settings.legend_spacing)

        self.img_exporter = ImageExporter(self.plotItem)
        self.svg_exporter = SVGExporter(self.plotItem)

        self.lr_item = None

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

    def plot(self, spectrum, **kwargs):
        """kwargs are passed to plotItem.plot function"""
        self.plotted_items.append(self.plotItem.plot(spectrum.data, **kwargs))
        self.plotted_spectra.append(spectrum)

    def remove(self, spectrum):
        try:
            i = self.plotted_spectra.index(spectrum)
        except ValueError:
            return

        self.removeItem(self.plotted_items[i])
        del self.plotted_spectra[i]
        del self.plotted_items[i]

    def clear_plots(self):
        """Removes all plots except of linear region item and ...."""
        self.plotted_spectra.clear()
        self.plotted_items.clear()
        self.plotItem.clearPlots()

    @classmethod
    def add_linear_region(cls, region=None, bounds=None, brush=None, orientation='vertical', z_value=-10):
        """boudns is tuple or None"""

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

        if in_scene:
            try:
                mousePoint = self.plotItem.vb.mapSceneToView(pos)
                n = Settings.coordinates_sig_figures
                self.coordinates_func(f"x={{:.{n}g}}, y={{:.{n}g}}".format(mousePoint.x(), mousePoint.y()))
            except:
                pass

        # self.label.setPos(mousePoint.x(), mousePoint.y())
        # self.label.setHtml("<span style='font-size: 26pt'>x={:01f}, <span style='font-size: 26pt'>y={:01f}".format(mousePoint.x(), mousePoint.y()))

        if in_scene and not in_legend:
            if self.viewport().cursor() != Qt.CrossCursor:
                # print("in view")
                self.viewport().setCursor(Qt.CrossCursor)

        if in_scene and in_legend:
            if self.viewport().cursor() != Qt.SizeAllCursor:
                # print("in legend")
                self.viewport().setCursor(Qt.SizeAllCursor)

    def add_legend(self, size=None, spacing=5, offset=(-30, 30)):
        self.legend = LegendItemModif(size, verSpacing=spacing, offset=offset)
        self.legend.setParentItem(self.plotItem.vb)
        self.plotItem.legend = self.legend
        # return self.legend
