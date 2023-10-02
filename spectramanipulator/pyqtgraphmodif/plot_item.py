# -*- coding: utf-8 -*-

# fcns addItem and plot were modified by adding an optional parameter plot_legend
# plot handled zValue kwarg

from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from pyqtgraph.graphicsItems.PlotItem import PlotItem as _PlotItem


class PlotItem(_PlotItem):

    def addItem(self, item, *args, **kargs):
        """
        Add a graphics item to the view box.
        If the item has plot data (:class:`PlotDataItem <pyqtgraph.PlotDataItem>` ,
        :class:`~pyqtgraph.PlotCurveItem` , :class:`~pyqtgraph.ScatterPlotItem` ),
        it may be included in analysis performed by the PlotItem.
        """
        if item in self.items:
            warnings.warn('Item already added to PlotItem, ignoring.')
            return
        self.items.append(item)
        vbargs = {}
        if 'ignoreBounds' in kargs:
            vbargs['ignoreBounds'] = kargs['ignoreBounds']
        self.vb.addItem(item, *args, **vbargs)
        name = None
        if hasattr(item, 'implements') and item.implements('plotData'):
            name = item.name()
            self.dataItems.append(item)
            # self.plotChanged()

            params = kargs.get('params', {})
            self.itemMeta[item] = params
            # item.setMeta(params)
            self.curves.append(item)
            # self.addItem(c)

        if hasattr(item, 'setLogMode'):
            item.setLogMode(self.ctrl.logXCheck.isChecked(), self.ctrl.logYCheck.isChecked())

        if isinstance(item, PlotDataItem):
            ## configure curve for this plot
            (alpha, auto) = self.alphaState()
            item.setAlpha(alpha, auto)
            item.setFftMode(self.ctrl.fftCheck.isChecked())
            item.setDownsampling(*self.downsampleMode())
            item.setClipToView(self.clipToViewMode())

            ## Hide older plots if needed
            self.updateDecimation()

            ## Add to average if needed
            self.updateParamList()
            if self.ctrl.averageGroup.isChecked() and 'skipAverage' not in kargs:
                self.addAvgCurve(item)

            # c.connect(c, QtCore.SIGNAL('plotChanged'), self.plotChanged)
            # item.sigPlotChanged.connect(self.plotChanged)
            # self.plotChanged()
        # name = kargs.get('name', getattr(item, 'opts', {}).get('name', None))
        plot_legend = kargs.get('plot_legend', True)  # added
        if name is not None and hasattr(self, 'legend') and self.legend is not None and plot_legend:
            self.legend.addItem(item, name=name)

    def plot(self, *args, **kargs):
        """
        Add and return a new plot.
        See :func:`PlotDataItem.__init__ <pyqtgraph.PlotDataItem.__init__>` for data arguments

        Extra allowed arguments are:
            clear    - clear all plots before displaying new data
            params   - meta-parameters to associate with this data
        """
        clear = kargs.get('clear', False)
        params = kargs.get('params', None)

        if clear:
            self.clear()

        item = PlotDataItem(*args, **kargs)

        if params is None:
            params = {}

        if 'zValue' in kargs:
            item.setZValue(kargs['zValue'])

        # plot legend option added
        self.addItem(item, params=params, plot_legend=kargs.get('plot_legend', True))

        return item














