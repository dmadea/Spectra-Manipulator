
from pyqtgraph.graphicsItems.ViewBox import ViewBox as _ViewBox

import math


def is_nan_or_inf(value):
    return math.isnan(value) or math.isinf(value)


class ViewBox(_ViewBox):
    # view All fix for downsampling ON

    def __init__(self, plotted_items: dict, **kwargs):
        super(ViewBox, self).__init__(**kwargs)

        self.plotted_items = plotted_items

    def autoRange(self, padding=None, items=None, item=None):
        """Modify the autorange function to properly autorange to all data."""

        spectra = list(self.plotted_items.keys())

        if len(spectra) == 0:
            return

        x0 = spectra[0].data[0, 0]
        x1 = spectra[0].data[-1, 0]

        y0 = spectra[0].y.min()
        y1 = spectra[0].y.max()

        for item in spectra[1:]:
            new_x0 = item.data[0, 0]
            new_x1 = item.data[-1, 0]
            new_y0 = item.y.min()
            new_y1 = item.y.max()

            if new_x0 < x0:
                x0 = new_x0

            if new_x1 > x1:
                x1 = new_x1

            if new_y0 < y0:
                y0 = new_y0

            if new_y1 > y1:
                y1 = new_y1

        x0 = -1 if is_nan_or_inf(x0) else x0
        x1 = 1 if is_nan_or_inf(x1) else x1
        y0 = -1 if is_nan_or_inf(y0) else y0
        y1 = 1 if is_nan_or_inf(y1) else y1

        self.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=padding)




