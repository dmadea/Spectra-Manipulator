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

""" Configuration tree. Can be used to manipulate the a ConfigurationModel.

"""
from __future__ import print_function

# import enum
import logging
# import os.path
# import sys

# from .abstractcti import ResetMode
from .configitemdelegate import ConfigItemDelegate
from .configtreemodel import ConfigTreeModel
# from argos.info import DEBUGGING, icons_directory
from PyQt5 import QtCore, QtWidgets
# from PyQt5.QtCore import Qt
from .argostreeview import ArgosTreeView
# from .constants import RIGHT_DOCK_WIDTH, DOCK_SPACING, DOCK_MARGIN
# from argos.widgets.misc import BasePanel
# from argos.utils.cls import check_class

RIGHT_DOCK_WIDTH = 200

logger = logging.getLogger(__name__)

# Qt classes have many ancestors
#pylint: disable=R0901


class ConfigTreeView(ArgosTreeView):
    """ Tree widget for manipulating a tree of configuration options.
    """
    def __init__(self, configTreeModel, parent=None):
        """ Constructor
        """
        super(ConfigTreeView, self).__init__(treeModel=configTreeModel, parent=parent)

        self._configTreeModel = configTreeModel

        self.expanded.connect(configTreeModel.expand)
        self.collapsed.connect(configTreeModel.collapse)
        #configTreeModel.update.connect(self.update) # not necessary
        #self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)

        treeHeader = self.header()
        treeHeader.resizeSection(ConfigTreeModel.COL_NODE_NAME, RIGHT_DOCK_WIDTH)
        treeHeader.resizeSection(ConfigTreeModel.COL_VALUE, RIGHT_DOCK_WIDTH)

        headerNames = self.model().horizontalHeaders
        enabled = dict((name, True) for name in headerNames)
        enabled[headerNames[ConfigTreeModel.COL_NODE_NAME]] = False # Name cannot be unchecked
        enabled[headerNames[ConfigTreeModel.COL_VALUE]] = False # Value cannot be unchecked
        checked = dict((name, False) for name in headerNames)
        checked[headerNames[ConfigTreeModel.COL_NODE_NAME]] = True # Checked by default
        checked[headerNames[ConfigTreeModel.COL_VALUE]] = True # Checked by default
        self.addHeaderContextMenu(checked=checked, enabled=enabled, checkable={})

        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setItemDelegate(ConfigItemDelegate())
        self.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)

        #self.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked |
        #                     QtWidgets.QAbstractItemView.EditKeyPressed |
        #                     QtWidgets.QAbstractItemView.AnyKeyPressed |
        #                     QtWidgets.QAbstractItemView.SelectedClicked)


    def sizeHint(self):
        """ The recommended size for the widget."""
        return QtCore.QSize(RIGHT_DOCK_WIDTH, 500)


    # @QtSlot(QtWidgets.QWidget, QtWidgets.QAbstractItemDelegate.EndEditHint)
    def closeEditor(self, editor, hint):
        """ Finalizes, closes and releases the given editor.
        """
        # It would be nicer if this method was part of ConfigItemDelegate since createEditor also
        # lives there. However, QAbstractItemView.closeEditor is sometimes called directly,
        # without the QAbstractItemDelegate.closeEditor signal begin emitted, e.g when the
        # currentItem changes. Therefore we cannot connect the QAbstractItemDelegate.closeEditor
        # signal to a slot in the ConfigItemDelegate.
        configItemDelegate = self.itemDelegate()
        configItemDelegate.finalizeEditor(editor)

        super(ConfigTreeView, self).closeEditor(editor, hint)


    def expandBranch(self, index=None, expanded=None):
        """ Expands or collapses the node at the index and all it's descendants.
            If expanded is True the nodes will be expanded, if False they will be collapsed, and if
            expanded is None the expanded attribute of each item is used.
            If parentIndex is None, the invisible root will be used (i.e. the complete forest will
            be expanded).
        """
        configModel = self.model()
        if index is None:
            #index = configTreeModel.createIndex()
            index = QtCore.QModelIndex()

        if index.isValid():
            if expanded is None:
                item = configModel.getItem(index)
                self.setExpanded(index, item.expanded)
            else:
                self.setExpanded(index, expanded)

        for rowNr in range(configModel.rowCount(index)):
            childIndex = configModel.index(rowNr, configModel.COL_NODE_NAME, parentIndex=index)
            self.expandBranch(index=childIndex, expanded=expanded)


    @property
    def autoReset(self):
        """ Indicates that the model will be (oartially) reset when the RTI or combo change
        """
        return self._configTreeModel.autoReset


    @autoReset.setter
    def autoReset(self, value):
        """ Indicates that the model will be (oartially) reset when the RTI or combo change
        """
        self._configTreeModel.autoReset = value


    @property
    def resetMode(self):
        """ Determines what is reset if autoReset is True (either axes or all settings)
        """
        return self._configTreeModel.resetMode


    @resetMode.setter
    def resetMode(self, value):
        """ Determines what is reset if autoReset is True (either axes or all settings)
        """
        self._configTreeModel.resetMode = value


    def resetAllSettings(self):
        """ Resets all settings
        """
        logger.debug("Resetting all settings")
        self._configTreeModel.resetAllSettings()


    def resetAllRanges(self):
        """ Resets all (axis/color/etc) range settings.
        """
        logger.debug("Resetting all range settings")
        self._configTreeModel.resetAllRanges()


