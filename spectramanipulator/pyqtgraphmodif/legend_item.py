
from pyqtgraph.graphicsItems.LegendItem import LegendItem as _LegendItem
from pyqtgraph.graphicsItems.LegendItem import ItemSample as _ItemSamle
from .label_item import LabelItem

from PyQt5 import QtGui, QtCore
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem, drawSymbol
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from pyqtgraph.graphicsItems.BarGraphItem import BarGraphItem

# changed legend line position in ItemSample


class LegendItem(_LegendItem):

    def __init__(self, size=None, offset=None, horSpacing=25, verSpacing=0,
                 pen=None, brush=None, labelTextColor=None, frame=True,
                 labelTextSize='9pt', rowCount=1, colCount=1,  **kwargs):

        super(LegendItem, self).__init__(size, offset, horSpacing, verSpacing,
                                         pen, brush, labelTextColor, frame,
                                         labelTextSize, rowCount, colCount, **kwargs)

        self.verSpacing = verSpacing  # verSpacing parameter used as a setup for setMinimumHeight in LabelItemModif

    def remove_last_item(self):
        sample, label = self.items[-1]
        del self.items[-1]
        self.layout.removeItem(sample)  # remove from layout
        sample.close()  # remove from drawing
        self.layout.removeItem(label)
        label.close()
        self.updateSize()  # redraq box

    def addItem(self, item, name):
        """
        Add a new entry to the legend.

        ==============  ========================================================
        **Arguments:**
        item            A :class:`~pyqtgraph.PlotDataItem` from which the line
                        and point style of the item will be determined or an
                        instance of ItemSample (or a subclass), allowing the
                        item display to be customized.
        title           The title to display for this item. Simple HTML allowed.
        ==============  ========================================================
        """
        # USED LabelItemModif insted of LabelItem
        label = LabelItem(name, color=self.opts['labelTextColor'],
                          justify='left', size=self.opts['labelTextSize'], verspacing=self.verSpacing)
        if isinstance(item, ItemSample):  # Changed from ItemSample to ItemSampleModif
            sample = item
        else:
            sample = ItemSample(item)  # Changed from ItemSample to ItemSampleModif
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()


class ItemSample(_ItemSamle):

    def paint(self, p, *args):
        opts = self.item.opts

        if opts.get('antialias'):
            p.setRenderHint(p.Antialiasing)

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            # p.drawLine(0, 11, 20, 11)
            p.drawLine(0, 15, 20, 15)  # CHANGED THIS LINE

            if (opts.get('fillLevel', None) is not None and
                    opts.get('fillBrush', None) is not None):
                p.setBrush(fn.mkBrush(opts['fillBrush']))
                p.setPen(fn.mkPen(opts['fillBrush']))
                p.drawPolygon(QtGui.QPolygonF(
                    [QtCore.QPointF(2, 18), QtCore.QPointF(18, 2),
                     QtCore.QPointF(18, 18)]))

        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts
            p.translate(10, 10)
            drawSymbol(p, symbol, opts['size'], fn.mkPen(opts['pen']),
                       fn.mkBrush(opts['brush']))

        if isinstance(self.item, BarGraphItem):
            p.setBrush(fn.mkBrush(opts['brush']))
            p.drawRect(QtCore.QRectF(2, 2, 18, 18))
