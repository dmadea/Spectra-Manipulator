import re
from pyqtgraph.exporters import Exporter
from pyqtgraph.exporters.Matplotlib import MatplotlibWindow as _MatplotlibWindow, _symbol_pg_to_mpl
from .plot_item import PlotItem
import pyqtgraph.functions as fn
from PyQt5 import QtGui, QtCore


def _strip_html(text):
    """Remove HTML tags from text (e.g., <strong>text</strong> -> text)."""
    if text is None or not isinstance(text, str):
        return text
    return re.sub(r'<[^>]+>', '', text)


class MatplotlibExporterModif(Exporter):
    Name = "Matplotlib Window legend"
    windows = []

    def __init__(self, item):
        Exporter.__init__(self, item)
        
    def parameters(self):
        return None

    def cleanAxes(self, axl):
        if type(axl) is not list:
            axl = [axl]
        for ax in axl:
            if ax is None:
                continue
            for loc, spine in ax.spines.items():
                if loc in ['left', 'bottom', 'right', 'top']:
                    pass  # show all four borders
                else:
                    raise ValueError('Unknown spine location: %s' % loc)
            # show ticks and labels on all four axes (mirror bottom/left on top/right)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='x', which='both', top=True, labeltop=True)
            ax.tick_params(axis='y', which='both', right=True, labelright=True)
    
    def export(self, fileName=None):
        if not isinstance(self.item, PlotItem):
            raise Exception("MatplotlibExporter currently only works with PlotItem")
    
        mpw = MatplotlibWindow()
        MatplotlibExporterModif.windows.append(mpw)

        fig = mpw.getFigure()

        xax = self.item.getAxis('bottom')
        yax = self.item.getAxis('left')
        
        # get labels from the graphic item (strip HTML)
        xlabel = _strip_html(xax.label.toPlainText())
        ylabel = _strip_html(yax.label.toPlainText())
        title = _strip_html(self.item.titleLabel.text)

        # if axes use autoSIPrefix, scale the data so mpl doesn't add its own
        # scale factor label
        xscale = yscale = 1.0
        if xax.autoSIPrefix:
            xscale = xax.autoSIPrefixScale
        if yax.autoSIPrefix:
            yscale = yax.autoSIPrefixScale

        ax = fig.add_subplot(111, title=title)
        ax.clear()
        self.cleanAxes(ax)
        has_legend_labels = False
        for item in self.item.curves:
            x, y = item.getData()
            x = x * xscale
            y = y * yscale

            opts = item.opts
            pen = fn.mkPen(opts['pen'])
            if pen.style() == QtCore.Qt.PenStyle.NoPen:
                linestyle = ''
            else:
                linestyle = '-'
            color = pen.color().getRgbF()
            symbol = opts['symbol']
            symbol = _symbol_pg_to_mpl.get(symbol, "")
            symbolPen = fn.mkPen(opts['symbolPen'])
            symbolBrush = fn.mkBrush(opts['symbolBrush'])
            markeredgecolor = symbolPen.color().getRgbF()
            markerfacecolor = symbolBrush.color().getRgbF()
            markersize = opts['symbolSize']

            # get item label for legend (same as used in pyqtgraph legend), strip HTML
            item_label = item.name() if hasattr(item, 'name') else opts.get('name')
            item_label = _strip_html(item_label) if item_label else item_label
            if item_label:
                has_legend_labels = True

            if opts['fillLevel'] is not None and opts['fillBrush'] is not None:
                fillBrush = fn.mkBrush(opts['fillBrush'])
                fillcolor = fillBrush.color().getRgbF()
                ax.fill_between(x=x, y1=y, y2=opts['fillLevel'], facecolor=fillcolor)
            
            ax.plot(x, y, marker=symbol, color=color, linewidth=pen.width(), 
                    linestyle=linestyle, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor,
                    markersize=markersize, label=item_label)

            xr, yr = self.item.viewRange()
            ax.set_xbound(xr[0]*xscale, xr[1]*xscale)
            ax.set_ybound(yr[0]*yscale, yr[1]*yscale)

        if has_legend_labels:
            ax.legend()

        ax.set_xlabel(xlabel)  # place the labels.
        ax.set_ylabel(ylabel)
        mpw.draw()
                
MatplotlibExporterModif.register()        


class MatplotlibWindow(_MatplotlibWindow):
        
    def closeEvent(self, ev):
        MatplotlibExporterModif.windows.remove(self)
        self.deleteLater()
