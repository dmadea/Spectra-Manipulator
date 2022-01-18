
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets
from config_sel.configtreemodel import ConfigTreeModel
from config_sel.configtreeview import ConfigTreeView
from config_sel.groupcti import MainGroupCti, GroupCti
from config_sel.intcti import IntCti
from config_sel.floatcti import FloatCti
from config_sel.stringcti import StringCti
from config_sel.boolcti import BoolCti, BoolGroupCti

from config_sel.qtctis import ColorCti, FontCti, PenCti

from PyQt5.QtGui import QPen



class Settings(object):

    # TODO ->  rewrite settings, redefine getter and setter and getitem, search and write by xpath

    plotting = {
        'type': 'group',
        'default_value': None,
        'str': 'string'
    }

    def __init__(self):
        pass




class TestCti(MainGroupCti):
    """ Configuration tree item for a PgImagePlot2dCti inspector
    """
    def __init__(self, nodeName):
        """ Constructor

            Maintains a link to the target pgImagePlot2d inspector, so that changes in the
            configuration can be applied to the target by simply calling the apply method.
            Vice versa, it can connect signals to the target.
        """
        super(TestCti, self).__init__(nodeName)

        self.insertChild(IntCti('title', 0, minValue=0, maxValue=100))

        test_group = GroupCti('test group')

        test_group.insertChild(IntCti('int1', 0, minValue=0, maxValue=100))
        test_group.insertChild(IntCti('int2', 5, minValue=0, maxValue=100))
        test_group.insertChild(IntCti('int3', 10, minValue=0, maxValue=100))

        self.insertChild(test_group)

        test_group2 = GroupCti('plotting')
        test_group2.insertChild(FloatCti('float1', 0.259, minValue=-1e5, maxValue=1e5))
        test_group2.insertChild(StringCti('string 2', "string"))
        test_group2.insertChild(BoolCti('bool test', True))

        self.insertChild(test_group2)

        x_axis = BoolGroupCti('x axis', True)
        x_axis.insertChild(StringCti('name', "Wavelength / nm"))
        x_axis.insertChild(BoolCti('show', True))
        x_axis.insertChild(BoolCti('show2', False))

        self.insertChild(x_axis)

        default_pen = pg.mkPen(color='blue', width=1, style=1)

        qt_group = GroupCti('qt_group')
        qt_group.insertChild(ColorCti('color', "blue"))
        qt_group.insertChild(FontCti('font', 'font'))
        qt_group.insertChild(PenCti('pen', False, resetTo=default_pen))

        self.insertChild(qt_group)


        # #### Axes ####
        #
        # self.aspectLockedCti = self.insertChild(PgAspectRatioCti(viewBox))
        #
        # self.xAxisCti = self.insertChild(PgAxisCti('x-axis'))
        # self.xAxisCti.insertChild(
        #     PgAxisLabelCti(imagePlotItem, 'bottom', self.pgImagePlot2d.collector,
        #                    defaultData=1,
        #                    configValues=[NO_LABEL_STR, "{x-dim} [index]"]))
        # self.xFlippedCti = self.xAxisCti.insertChild(PgAxisFlipCti(viewBox, X_AXIS))
        # self.xAxisRangeCti = self.xAxisCti.insertChild(PgAxisRangeCti(viewBox, X_AXIS))

        # self.yAxisCti = self.insertChild(PgAxisCti('y-axis'))
        # self.yAxisCti.insertChild(
        #     PgAxisLabelCti(imagePlotItem, 'left', self.pgImagePlot2d.collector,
        #                    defaultData=1,
        #                    configValues=[NO_LABEL_STR, "{y-dim} [index]"]))
        # self.yFlippedCti = self.yAxisCti.insertChild(
        #     PgAxisFlipCti(viewBox, Y_AXIS, defaultData=True))
        # self.yAxisRangeCti = self.yAxisCti.insertChild(PgAxisRangeCti(viewBox, Y_AXIS))
        #
        # #### Color scale ####
        #
        # self.colorCti = self.insertChild(PgAxisCti('color scale'))
        #
        # self.colorCti.insertChild(PgColorLegendLabelCti(
        #     pgImagePlot2d.colorLegendItem, self.pgImagePlot2d.collector, defaultData=1,
        #     configValues=[NO_LABEL_STR, "{name} {unit}", "{path} {unit}",
        #                   "{name}", "{path}", "{raw-unit}"]))
        #
        # self.colorCti.insertChild(PgColorMapCti(self.pgImagePlot2d.colorLegendItem))
        #
        # self.showHistCti = self.colorCti.insertChild(
        #     PgShowHistCti(pgImagePlot2d.colorLegendItem))
        # self.showDragLinesCti = self.colorCti.insertChild(
        #     PgShowDragLinesCti(pgImagePlot2d.colorLegendItem))
        #
        # colorAutoRangeFunctions = defaultAutoRangeMethods(self.pgImagePlot2d)
        # self.colorLegendCti = self.colorCti.insertChild(
        #     PgColorLegendCti(pgImagePlot2d.colorLegendItem, colorAutoRangeFunctions,
        #                      nodeName="range"))
        #
        # # If True, the image is automatically downsampled to match the screen resolution. This
        # # improves performance for large images and reduces aliasing. If autoDownsample is not
        # # specified, then ImageItem will choose whether to downsample the image based on its size.
        # self.autoDownSampleCti = self.insertChild(BoolCti('auto down sample', True))
        # self.zoomModeCti = self.insertChild(BoolCti('rectangle zoom mode', False))
        #
        # ### Probe and cross-hair plots ###
        #
        # self.probeCti = self.insertChild(BoolCti('show probe', True))
        # self.crossPlotGroupCti = self.insertChild(BoolGroupCti('cross-hair', expanded=False))
        # self.crossPenCti = self.crossPlotGroupCti.insertChild(PgPlotDataItemCti(expanded=False))
        #
        # self.horCrossPlotCti = self.crossPlotGroupCti.insertChild(
        #     BoolCti('horizontal', False, expanded=False))
        #
        # self.horCrossPlotCti.insertChild(PgGridCti(pgImagePlot2d.horCrossPlotItem))
        # self.horCrossPlotRangeCti = self.horCrossPlotCti.insertChild(
        #     PgAxisRangeCti(
        #         self.pgImagePlot2d.horCrossPlotItem.getViewBox(), Y_AXIS, nodeName="data range",
        #         autoRangeFunctions=crossPlotAutoRangeMethods(self.pgImagePlot2d, "horizontal")))
        #
        # self.verCrossPlotCti = self.crossPlotGroupCti.insertChild(
        #     BoolCti('vertical', False, expanded=False))
        # self.verCrossPlotCti.insertChild(PgGridCti(pgImagePlot2d.verCrossPlotItem))
        # self.verCrossPlotRangeCti = self.verCrossPlotCti.insertChild(
        #     PgAxisRangeCti(
        #         self.pgImagePlot2d.verCrossPlotItem.getViewBox(), X_AXIS, nodeName="data range",
        #         autoRangeFunctions=crossPlotAutoRangeMethods(self.pgImagePlot2d, "vertical")))

        # Connect signals.

        # Use a queued connect to schedule the reset after current events have been processed.
        # self.pgImagePlot2d.colorLegendItem.sigResetColorScale.connect(
        #     self.colorLegendCti.setAutoRangeOn, type=Qt.QueuedConnection)
        # self.pgImagePlot2d.imagePlotItem.sigResetAxis.connect(
        #     self.setImagePlotAutoRangeOn, type=Qt.QueuedConnection)
        # self.pgImagePlot2d.horCrossPlotItem.sigResetAxis.connect(
        #     self.setHorCrossPlotAutoRangeOn, type=Qt.QueuedConnection)
        # self.pgImagePlot2d.verCrossPlotItem.sigResetAxis.connect(
        #     self.setVerCrossPlotAutoRangeOn, type=Qt.QueuedConnection)
        #
        # # Also update axis auto range tree items when linked axes are resized
        # horCrossViewBox = self.pgImagePlot2d.horCrossPlotItem.getViewBox()
        # horCrossViewBox.sigRangeChangedManually.connect(self.xAxisRangeCti.setAutoRangeOff)
        # verCrossViewBox = self.pgImagePlot2d.verCrossPlotItem.getViewBox()
        # verCrossViewBox.sigRangeChangedManually.connect(self.yAxisRangeCti.setAutoRangeOff)



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    tm = ConfigTreeModel()
    tv = ConfigTreeView(tm)

    mg_cti = TestCti('main')
    tm.setInvisibleRootItem(mg_cti)
    tv.expandBranch()



    tv.show()
    sys.exit(app.exec())

