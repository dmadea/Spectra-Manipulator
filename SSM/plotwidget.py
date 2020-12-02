

from PyQt5 import QtGui
import pyqtgraph as pg

from .pyqtgraphmodif.legend_item_modif import LegendItemModif
from .pyqtgraphmodif.plot_item_modif import PlotItemModif

from pyqtgraph.exporters import ImageExporter
from pyqtgraph.exporters import SVGExporter
from .settings import Settings

from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QColor


# subclassing of PlotWidget
class PlotWidget(pg.PlotWidget):
    instance = None

    def __init__(self, parent=None, coordinates_func=None):

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        super(PlotWidget, self).__init__(parent, plotItem=PlotItemModif())

        PlotWidget.instance = self

        self.coordinates_func = coordinates_func

        self.update_settings()

        self.legend = None
        self.add_legend()

        self.img_exporter = ImageExporter(self.plotItem)
        self.svg_exporter = SVGExporter(self.plotItem)

    def update_settings(self):

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
