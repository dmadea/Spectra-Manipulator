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

""" Argos Color Library
"""
from __future__ import print_function

import logging
import os.path

from os import listdir

from cmlib import CmLib, CmLibModel, DATA_DIR

from ..singleton import Singleton

logger = logging.getLogger(__name__)

# The color maps that are favorites then the program is started for the first time or reset.
DEF_FAV_COLOR_MAPS = [
    'MatPlotLib/Gray', 'MatPlotLib/Hsv', 'MatPlotLib/Hot', 'MatPlotLib/Magma',
    'MatPlotLib/Viridis', 'MatPlotLib/Inferno', 'MatPlotLib/Jet', 'MatPlotLib/Seismic',
    'MatPlotLib/Oranges']


DEFAULT_COLOR_MAP = "MatPlotLib/Hsv"
assert DEFAULT_COLOR_MAP in DEF_FAV_COLOR_MAPS, "Default color map not in default favorites."


class CmLibSingleton(CmLib, Singleton):

    def __init__(self, **kwargs):
        super(CmLibSingleton, self).__init__(**kwargs)

        logger.debug("CmLib singleton: {}".format(self))

        cmDataDir = DATA_DIR
        logger.info("Importing color map library from: {}".format(cmDataDir))

        # Don't import from Color Brewer since those are already included in MatPlotLib.
        # With sub-sampling the color maps similar maps can be achieved as the Color Brewer maps.
        excludeList = ['ColorBrewer2']
        for path in listdir(cmDataDir):
            if path in excludeList:
                logger.debug("Not importing catalogue from exlude list: {}".format(excludeList))
                continue

            fullPath = os.path.join(cmDataDir, path)
            if os.path.isdir(fullPath):
                self.load_catalog(fullPath)

        logger.debug("Number of color maps: {}".format(len(self.color_maps)))

        for colorMap in self.color_maps:
            # print(colorMap)
            colorMap.meta_data.favorite = colorMap.key in DEF_FAV_COLOR_MAPS


class CmLibModelSingleton(CmLibModel, Singleton):

    def __init__(self, **kwargs):
        cmlib = CmLibSingleton()
        super(CmLibModelSingleton, self).__init__(cmlib, **kwargs)

