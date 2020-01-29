from PyQt5 import QtGui


import pyqtgraph as pg
from settings import Settings

from pyqtgraphmodif.legend_item import LegendItem

from pyqtgraph.exporters import ImageExporter
from pyqtgraph.exporters import SVGExporter

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


class PlotWidget(pg.PlotWidget):
    instance = None

    def __init__(self, parent=None, coordinates_func=None):

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        super(PlotWidget, self).__init__(parent)

        PlotWidget.instance = self

        # self.label = pg.TextItem("text", color=(0, 0, 0))
        # self.addItem(self.label)

        # self.view().setLeftButtonAction('pan')

        self.plt = self.plotItem

        self.coordinates_func = coordinates_func

        self.update_settings()

        self.legend = None
        self.add_legend()

        # proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def update_settings(self):

        pg.setConfigOptions(antialias=Settings.antialiasing)
        self.plt.showAxis('top', show=True)
        self.plt.showAxis('right', show=True)

        self.plt.setTitle(Settings.graph_title, size="{}pt".format(Settings.graph_title_font_size))

        self.plt.setLabel('left', text=Settings.left_axis_label, units=Settings.left_axis_unit)
        self.plt.setLabel('bottom', text=Settings.bottom_axis_label, units=Settings.bottom_axis_unit)
        self.plt.showGrid(x=Settings.show_grid, y=Settings.show_grid, alpha=Settings.grid_alpha)

        left_label_font = QtGui.QFont()
        left_label_font.setPixelSize(Settings.left_axis_font_size)

        self.plt.getAxis('left').label.setFont(left_label_font)

        bottom_label_font = QtGui.QFont()
        bottom_label_font.setPixelSize(Settings.bottom_axis_font_size)

        self.plt.getAxis('bottom').label.setFont(bottom_label_font)

    def save_plot_to_clipboard_as_png(self):

        self.img_exporter = ImageExporter(self.plotItem)

        # set export parameters if needed - nefunguje tak jak bych chtel
        # img_exporter.parameters()['width'] = 10000  # (note this also affects height parameter)
        # img_exporter.parameters()['background'] = QColor(255, 255, 255, 0)

        self.img_exporter.export(copy=True)

    def save_plot_to_clipboard_as_svg(self):

        self.svg_exporter = SVGExporter(self.plotItem)
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
                self.coordinates_func("x={:4.4g}, y={:4.4g}".format(mousePoint.x(), mousePoint.y()))
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

    # def mousePressEvent(self, ev):
    #
    #     # ev.doubleClick
    #     super(PlotWidget, self).mousePressEvent(ev)
    #     # print("mouse pressed", ev.pos())
    #
    #
    # def mouseReleaseEvent(self, ev):
    #     super(PlotWidget, self).mouseReleaseEvent(ev)
    #     # print("mouse released", ev.pos())

    def add_legend(self, size=None, spacing=5, offset=(-30, 30)):
        # self.legend = LegendItem(size, spacing, offset)
        self.legend = LegendItem(size, spacing, offset)
        self.legend.setParentItem(self.plotItem.vb)
        self.plotItem.legend = self.legend
        # return self.legend
