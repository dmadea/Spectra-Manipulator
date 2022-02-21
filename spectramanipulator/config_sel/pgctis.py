# -*- coding: utf-8 -*-

# This file is part of Argos.
#
# Argos is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Argos is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Argos. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division, print_function

import logging
import numpy as np
import pyqtgraph as pg

# Using partial as Python closures are late-binding.
# http://docs.python-guide.org/en/latest/writing/gotchas/#late-binding-closures
# from functools import partial
# from collections import OrderedDict

from cmlib import ColorSelectionWidget, ColorMap, makeColorBarPixmap
from cmlib import CmMetaData, CatalogMetaData

from PyQt5.QtWidgets import QSizePolicy

from .groupcti import GroupCti
from .abstractcti import AbstractCti, AbstractCtiEditor
# from .boolcti import BoolCti
# from .intcti import IntCti
from .misc import setWidgetSizePolicy
from .colors import CmLibModelSingleton, DEFAULT_COLOR_MAP

logger = logging.getLogger(__name__)
# logger = logging

X_AXIS = pg.ViewBox.XAxis
Y_AXIS = pg.ViewBox.YAxis
BOTH_AXES = pg.ViewBox.XYAxes
VALID_AXIS_NUMBERS = (X_AXIS, Y_AXIS, BOTH_AXES)
VALID_AXIS_POSITIONS = ('left', 'right', 'bottom', 'top')
NO_LABEL_STR = '-- none --'


class PgColorMapCti(GroupCti):
    """ Lets the user select one of color maps of the color map library
    """
    SUB_SAMPLING_OFF = 1

    def __init__(self, nodeName="color map", defaultData=DEFAULT_COLOR_MAP, expanded=True, **kwargs):
        """ Constructor.

            Stores a color map as data.

            :param defaultData: the default index in the combobox that is used for editing
        """
        # self.colorLegendItem = colorLegendItem
        self.cmLibModel = CmLibModelSingleton()

        # grey scale color map for when no color map is selected.
        lutRgba = np.outer(np.arange(256, dtype=np.uint8), np.array([1, 1, 1, 1], dtype=np.uint8))
        lutRgba[:, 3] = 255
        self.greyScaleColorMap = ColorMap(CmMetaData("-- none --"), CatalogMetaData("Argos"))
        self.greyScaleColorMap.set_rgba_uint8_array(lutRgba)

        super(PgColorMapCti, self).__init__(nodeName, defaultData, expanded=expanded, **kwargs)

    def _enforceDataType(self, data):
        """ Converts to int so that this CTI always stores that type.
        """
        if data is None:
            return self.greyScaleColorMap
        elif isinstance(data, ColorMap):
            return data
        else:
            return self.cmLibModel.getColorMapByKey(data)

    @property
    def configValue(self):
        """ The currently selected configValue

            :rtype: ColorMap
        """
        return self.data

    # def _updateTargetFromNode(self):
    #     """ Applies the configuration to its target axis.
    #         Sets the image item's lookup table to the LUT of the selected color map.
    #     """
    #     lut = self.data.rgb_uint8_array
    #
    #     targetSize = self.subSampleCti.configValue
    #     if targetSize > self.SUB_SAMPLING_OFF:
    #         sourceSize, _ = lut.shape
    #         subIdx = np.round(np.linspace(0, sourceSize-1, targetSize)).astype(np.uint)
    #         lut = lut[subIdx, :]
    #
    #     if self.reverseCti.configValue:
    #         lut = np.flipud(lut)
    #
    #     # self.colorLegendItem.setLut(lut)

    def _dataToString(self, data):
        """ Conversion function used to convert the (default)data to the display value.
        """
        return "" if data is None else data.meta_data.pretty_name

    @property
    def decoration(self):
        """ Returns a pixmap of the color map to show as icon
        """
        return makeColorBarPixmap(
            self.data,
            width=self.cmLibModel.iconBarWidth * 0.65,
            height=self.cmLibModel.iconBarHeight * 0.65,
            drawBorder=self.cmLibModel.drawIconBarBorder)

    def _nodeMarshall(self):
        """ Returns the non-recursive marshalled value of this CTI. Is called by marshall()
        """
        return self.data.key

    def _nodeUnmarshall(self, key):
        """ Initializes itself non-recursively from data. Is called by unmarshall()
        """
        self.data = self.cmLibModel.getColorMapByKey(key)

    def createEditor(self, delegate, parent, option):
        """ Creates a ChoiceCtiEditor.
            For the parameters see the AbstractCti constructor documentation.
        """
        return PgColorMapCtiEditor(self, delegate, parent=parent)


class PgColorMapCtiEditor(AbstractCtiEditor):
    """ A CtiEditor which contains a QCombobox for editing ChoiceCti objects.
    """
    def __init__(self, cti, delegate, parent=None):
        """ See the AbstractCtiEditor for more info on the parameters
        """
        super(PgColorMapCtiEditor, self).__init__(cti, delegate, parent=parent)

        selectionWidget = ColorSelectionWidget(self.cti.cmLibModel)

        # need to uncheck recommended checkbox in Filter dialog, because cmlib takes
        # the colormaps only from recommended list...
        cb_recommended = selectionWidget.browser.filterForm._defaultOnCheckboxes[0]
        cb_recommended.setChecked(False)

        setWidgetSizePolicy(selectionWidget, QSizePolicy.Expanding, None)

        selectionWidget.sigColorMapHighlighted.connect(self.onColorMapHighlighted)
        selectionWidget.sigColorMapChanged.connect(self.onColorMapChanged)
        self.selectionWidget = self.addSubEditor(selectionWidget, isFocusProxy=True)
        self.comboBox = self.selectionWidget.comboBox

    def finalize(self):
        """ Is called when the editor is closed. Disconnect signals.
        """
        logger.debug("PgColorMapCtiEditor.finalize")
        self.selectionWidget.sigColorMapChanged.disconnect(self.onColorMapChanged)
        self.selectionWidget.sigColorMapHighlighted.disconnect(self.onColorMapHighlighted)
        super(PgColorMapCtiEditor, self).finalize()

    def setData(self, data):
        """ Provides the main editor widget with a data to manipulate.
        """
        if data is None:
            logger.warning("No color map to select")
        else:
            self.selectionWidget.setColorMapByKey(data.key)

    def getData(self):
        """ Gets data from the editor widget.
        """
        return self.selectionWidget.getCurrentColorMap()

    def onColorMapHighlighted(self, colorMap):
        """ Is called when the user highlights an item in the combo box or dialog.

            The item's index is passed.
            Note that this signal is sent even when the choice is not changed.
        """
        logger.debug("onColorMapHighlighted({})".format(colorMap))
        self.cti.data = colorMap
        self.cti.updateTarget()

    def onColorMapChanged(self, index):
        """ Is called when the user chooses an item in the combo box. The item's index is passed.
            Note that this signal is sent even when the choice is not changed.
        """
        logger.debug(f"onColorMapChanged, {self.cti.data}")
        self.delegate.commitData.emit(self)

