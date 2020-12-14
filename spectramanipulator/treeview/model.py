from abc import abstractmethod
import sys
from sys import platform
from copy import deepcopy

from PyQt5.QtWidgets import QTreeView, QAbstractItemView, QStyledItemDelegate
from PyQt5.QtCore import pyqtSignal, QAbstractItemModel, QVariant, QModelIndex, QItemSelection, QItemSelectionModel, Qt
from PyQt5.QtGui import QFont

from .pymimedata import PyMimeData
from .item import GenericItem, SpectrumItemGroup, SpectrumItem

# from spectramanipulator import Spectrum, SpectrumList

import cProfile
import pstats

import numpy as np

# possible workaround - https://stackoverflow.com/questions/34419072/how-to-improve-selection-performance-with-pyside-qtcore-qabstractitemmodel-and-q

# this was tough to get together, I could not use the TreeWidget with StandardItemModel,
# because iteration over items sometime gave not my item (SpectrumItem or SpectrumItemGroup) with my data,
# but the blank TreeWidgetItem, which raised errors and that sucked a lot, therefore, only solution was to
# implement QAbstractItemModel and then use TreeView instead of TreeWidget
# at least, now, it really works, but I had to write everything by myself... great thing about it is moving the
# items with data in between two separate opened applications, it would be almost impossible to do that in
# QTreeWidget...
# eventually, it is slower than QTreeWidget, but it is just python and Qt, I cannot expect miracles from it

# I got help from many sources...

# https://stackoverflow.com/questions/841096/slow-selection-in-qtreeview-why
# https://stackoverflow.com/questions/8175122/qtreeview-checkboxes
# https://riverbankcomputing.com/pipermail/pyqt/2009-April/022729.html
# http://flame-blaze.net/archives/5249
# http://doc.qt.io/archives/qt-4.8/qstandarditemmodel.html


class ItemIterator:
    NoChildren = 0
    Checked = 1
    All = 2
    Groups = 3


class Model(QAbstractItemModel):
    checked_changed_signal = pyqtSignal()
    data_dropped_signal = pyqtSignal()
    item_edited_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(Model, self).__init__(parent)

        self.treeView = parent
        self.headers = ['Name', 'Index | Info']

        self.columns = 2

        self.coarsed_mimeData = None

        # Create items
        self.root = GenericItem('root', 'this is root', None)

        # itemA = SpectrumItemGroup('itemA', 'this is item A', self.root)
        # itemA1 = SpectrumItem(None, 'itemA1', 'this is item A1', itemA)
        # #
        # itemB = SpectrumItemGroup('itemB', 'this is item B', self.root)
        # itemB1 = SpectrumItem(None, '1', 'this is item B1', itemB)
        # itemB2 = SpectrumItem(None, '2', 'this is item B1', itemB)
        # itemB3 = SpectrumItem(None, '3', 'this is item B1', itemB)
        #
        # itemC = SpectrumItemGroup('itemC', 'this is item C', self.root)
        # itemC1 = SpectrumItem(None, 'itemC1', 'this is item C1', self.root)

    def iterate_items(self, what_items):
        """
        Iterate specified items of TreeView.

        :param what_items: ItemIterator
        """
        if what_items == ItemIterator.Groups:
            for item in self.root.children:
                if isinstance(item, SpectrumItemGroup):
                    yield item
        elif what_items == ItemIterator.All:
            for item in self.root.children:
                yield item
                for child in item.children:
                    yield child
        elif what_items == ItemIterator.Checked:
            for item in self.root.children:
                if item.__len__() == 0:
                    if item.check_state == Qt.Checked or item.check_state == Qt.PartiallyChecked:
                        yield item
                        continue
                for child in item.children:
                    if child.check_state == Qt.Checked:
                        yield child
        elif what_items == ItemIterator.NoChildren:
            for item in self.root.children:
                if item.__len__() == 0:
                    yield item
                for child in item.children:
                    if child.__len__() == 0:
                        yield child

    def iterate_selected_items(self, skip_groups=False, skip_childs_in_selected_groups=True,
                               clear_selection=False):

        indexes = self.treeView.selectedIndexes()

        if len(indexes) == 0:
            return

        selected_items = []

        # step is two, because we have two columns, so there is 2 indexes per one item
        # first we have to sort the selected items because the are not sorted...
        for i in range(0, len(indexes), 2):
            idx = indexes[i]
            item = self.node_from_index(indexes[i])

            order = int(100000000)
            if item.is_top_level():
                # ix = self.indexFromItem(item)
                # row = idx.row()
                # print(row)
                order *= (idx.row() + 1)
            else:
                order *= (self.root.rowOfChild(item.parent) + 1)
                order += idx.row() + 1
            selected_items.append((item, order))

        selected_items = sorted(selected_items, key=lambda x: x[1])
        sorted_sel_items = []
        for item in selected_items:
            sorted_sel_items.append(item[0])

        if clear_selection:
            self.treeView.clearSelection()

        last_group = None
        for item in sorted_sel_items:
            if isinstance(item, SpectrumItemGroup):
                last_group = item
                if not skip_groups:
                    yield item
            if isinstance(item, SpectrumItem):
                # skip the child if only iterating over childs, whose parents are not selected
                if skip_childs_in_selected_groups and item.parent == last_group:
                    continue
                yield item

    def supportedDropActions(self):
        return Qt.MoveAction | Qt.CopyAction

    def flags(self, index):
        defaultFlags = QAbstractItemModel.flags(self, index)

        if index.isValid():

            if index.column() == 0:
                return Qt.ItemIsEditable | Qt.ItemIsUserCheckable | Qt.ItemIsTristate | Qt.ItemIsDragEnabled | \
                       Qt.ItemIsDropEnabled | defaultFlags
            else:
                return Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled | defaultFlags

        else:
            return Qt.ItemIsDropEnabled | defaultFlags

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headers[section])
        return QVariant()

    def mimeTypes(self):
        return [PyMimeData.MIME_TYPE]

    def mimeData(self, Iterable, q_model_index=None):

        item_list = [item for item in self.iterate_selected_items(False, True)]

        mimeData = PyMimeData(item_list)
        return mimeData

    def canDropMimeData(self, mimedata, action, row, column, parentIndex):
        parentNode = self.node_from_index(parentIndex)
        # print(action, row, column, parentNode.name)
        if self.coarsed_mimeData is None and not isinstance(mimedata, PyMimeData):

            if mimedata.hasUrls():
                for url in mimedata.urls():
                    if url.isLocalFile():
                        return True
                return False
            if mimedata.hasText():
                return True
            if mimedata.hasFormat("XML Spreadsheet"):
                return True
            if mimedata.hasFormat(PyMimeData.MIME_TYPE):
                mimedata = PyMimeData.coerce(mimedata)
                self.coarsed_mimeData = mimedata
            return False

        if not isinstance(mimedata, PyMimeData) and self.coarsed_mimeData is not None:
            mimedata = self.coarsed_mimeData
        # else:
        #     raise Exception("canDropMimeData - MimeData is not type of PyMimeData")

        dropped_data = mimedata.instance()

        if isinstance(parentNode, SpectrumItem):
            return False

        if parentNode.is_root():
            return True

        contain_group = False
        for item in dropped_data:
            if isinstance(item, SpectrumItemGroup):
                contain_group = True
                break

        if contain_group:
            return False

        return True

    def dropMimeData(self, mimedata, action, row, column, parentIndex):
        if not isinstance(mimedata, PyMimeData):
            mimedata = PyMimeData.coerce(mimedata)

        if action == Qt.IgnoreAction:
            return True

        dragNodes = mimedata.instance()
        parentNode = self.node_from_index(parentIndex)

        row_to_insert = len(parentNode) if row == -1 else row

        # make an copy of the node being moved
        newNodes = deepcopy(dragNodes)
        for node in reversed(newNodes):
            node.setParent(parentNode, row=row_to_insert)

        self.insertRows(row_to_insert, len(dragNodes), parentIndex)

        self._update_tristate(parentNode)
        self.dataChanged.emit(parentIndex, parentIndex)
        self.data_dropped_signal.emit()
        # self.emit(SIGNAL("dataChanged(QModelIndex,QModelIndex)"),
        #           parentIndex, parentIndex)

        self.coarsed_mimeData = None
        return True

    def take_all_children(self, group_item):
        parent_index = self.index_from_node(group_item)
        self.beginRemoveRows(parent_index, 0, len(group_item.children) - 1)
        ref_children_list = group_item.children
        group_item.children = []  # reassign to empty list
        self.endRemoveRows()
        return ref_children_list

    def take_item(self, item):
        parent_index = self.index_from_node(item.parent)
        row = item.parent.rowOfChild(item)
        self.beginRemoveRows(parent_index, row, row)
        item.parent.removeChildAtRow(row)
        self.endRemoveRows()
        # item.setParent(None)
        return item

    # def move_items(self, items):
    #
    #     source_parent = self.myModel.index_from_node(last_parent)
    #
    #     self.myModel.beginMoveRows(source_parent, first_row, last_row, group_item_index, i)
    #
    #     chunk = last_parent.children[first_row:last_row + 1]
    #     del last_parent.children[first_row:last_row + 1]
    #
    #     for child in chunk:
    #         child.setParent(group_item)
    #
    #     # item.parent.removeChildAtRow(row)
    #     # item.setParent(group_item)
    #
    #     self.myModel.endMoveRows()

    def move_item(self, item, destination_group, row_to_place=None):

        destination_parent = self.index_from_node(destination_group)
        source_parent = self.index_from_node(item.parent)

        if item.parent == destination_group:
            destination_parent = source_parent

        row = item.parent.rowOfChild(item)

        self.beginMoveRows(source_parent, row, row, destination_parent,
                           0 if row_to_place is None else row_to_place)

        if item.parent != destination_group:
            item.parent.removeChildAtRow(row)
            item.setParent(destination_group)
        else:
            destination_group.move_child(row, row_to_place)

        self.endMoveRows()

    def add_items(self, items, parent_item, row=None):
        for item in reversed(items):
            item.setParent(parent_item, row)
        self.insertRows(row if row is not None else len(parent_item.children), len(items),
                        self.index_from_node(parent_item))

    def add_item(self, item, parent_item, row=None):
        """Add child to group a correspondingly updates the view, if row=None, child will be appended at the end."""
        item.setParent(parent_item, row)
        self.insertRow(row if row is not None else len(parent_item.children), self.index_from_node(parent_item))

    def insertRow(self, row, parentIndex):
        return self.insertRows(row, 1, parentIndex)

    def insertRows(self, row, count, parentIndex):
        self.beginInsertRows(parentIndex, row, (row + count - 1))
        self.endInsertRows()
        return True

    def removeRow(self, row, parentIndex):
        return self.removeRows(row, 1, parentIndex)

    def removeRows(self, row, count, parentIndex, update_tristate=True):
        self.beginRemoveRows(parentIndex, row, (row + (count - 1)))
        node = self.node_from_index(parentIndex)
        # print(count)
        del node.children[row:row + count]

        # for i in range(count):
        #     node.removeChildAtRow(row)
        self.endRemoveRows()

        if update_tristate:
            self.update_tristate()
            self.treeView.setup_info()

        return True

    def index(self, row, column, parent):
        node = self.node_from_index(parent)
        return self.createIndex(row, column, node.childAtRow(row))

    def index_from_node(self, node):
        if node is None:
            return QModelIndex()

        parent = node.parent
        return QModelIndex() if parent is None else self.createIndex(parent.rowOfChild(node), 0, node)
        # return self.createIndex(parent.rowOfChild(node) if parent is not None else 0, 0, node)

    def data(self, index, role=None):

        # print("data")

        if role == Qt.DecorationRole:
            return QVariant()

        if role == Qt.TextAlignmentRole:
            return QVariant(int(Qt.AlignTop | Qt.AlignLeft))

        node = self.node_from_index(index)

        if node is None:
            return QVariant()

        if role == Qt.FontRole:
            font = QFont()
            if isinstance(node, SpectrumItemGroup):
                font.setBold(True)

            return QVariant(font)

        if role == Qt.CheckStateRole and index.column() == 0:
            return QVariant(node.check_state)

        if role != Qt.DisplayRole:
            return QVariant()

        if index.column() == 0:
            return QVariant(node.name)

        elif index.column() == 1:
            return QVariant(node.info)

        else:
            return QVariant()

    def columnCount(self, parent):
        return self.columns

    def rowCount(self, parent):
        node = self.node_from_index(parent)
        if node is None:
            return 0
        return len(node)

    def parent(self, index=None):
        if not index.isValid():
            return QModelIndex()

        node = self.node_from_index(index)

        if node is None:
            return QModelIndex()

        parent = node.parent

        if parent is None:
            return QModelIndex()

        grandparent = parent.parent
        if grandparent is None:
            return QModelIndex()
        # print("parent")
        row = grandparent.rowOfChild(parent)

        # row = 0

        # assert row != - 1
        return self.createIndex(row, 0, parent)

    def update_tristate(self):
        for group in self.iterate_items(ItemIterator.Groups):
            self._update_tristate(group)

    def _update_tristate(self, parent):

        if not isinstance(parent, SpectrumItemGroup):
            return

        checked_items = 0
        for item in parent.children:
            if item.isChecked():
                checked_items += 1
        if checked_items == 0:
            parent.check_state = Qt.Unchecked
        elif checked_items < parent.__len__():
            parent.check_state = Qt.PartiallyChecked
        else:
            parent.check_state = Qt.Checked

        idx = self.createIndex(parent.row(), 0, parent)

        self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])
        self.checked_changed_signal.emit()

    def setData(self, index, value, role=None):
        if index.column() == 0:
            if role == Qt.EditRole:
                node = self.node_from_index(index)
                node.name = value
                self.dataChanged.emit(index, index, [Qt.EditRole])
                self.item_edited_signal.emit(node.isChecked())
                return True

            # this part of code deals with just setting a proper check state of item
            # unchecked, checked and partially checked...
            if role == Qt.CheckStateRole:
                # print("role == Qt.CheckStateRole:")
                node = self.node_from_index(index)
                node.check_state = value

                if isinstance(node, SpectrumItemGroup):
                    if node.__len__() != 0:
                        for item in node.children:
                            item.check_state = value

                        # Index of last child in group
                        idx1 = self.createIndex(node.__len__() - 1, 0, node.children[-1])

                        self.dataChanged.emit(index, idx1, [Qt.CheckStateRole])
                    else:
                        self.dataChanged.emit(index, index, [Qt.CheckStateRole])

                    # self.checked_changed_signal.emit()

                elif isinstance(node, SpectrumItem):  # SpectrumItem instance
                    # print("isinstance(node, SpectrumItem):")

                    if node.is_in_group():
                        self._update_tristate(node.parent)

                self.checked_changed_signal.emit()
                return True

                # return super(Model, self).setData(index, value, role)

        return super(Model, self).setData(index, value, role)

    def node_from_index(self, index):
        return index.internalPointer() if index.isValid() else self.root


# this is just to keep previous text in field when editing
# http://www.informit.com/articles/article.aspx?p=1405547&seqNum=4
class ItemDelegate(QStyledItemDelegate):

    def setEditorData(self, lineEdit, index):
        previous_text = str(index.model().data(index, Qt.DisplayRole).value())
        lineEdit.setText(previous_text)


class TreeView(QTreeView):

    selecting = False  # this has to be static, method selectionChanged is called before __init__, wtf ??
    deleting = False

    items_deleted_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(TreeView, self).__init__(parent)

        self.myModel = Model(self)
        self.setModel(self.myModel)

        self.setSelectionMode(self.ExtendedSelection)
        self.setSelectionBehavior(QTreeView.SelectRows)
        self.setItemDelegateForColumn(0, ItemDelegate(self))

        self.setUniformRowHeights(True)

        # this is the most important line of code here, this is a result of many hours of unsuccessful tries
        # this sets MoveAction as a default DropAction while dragging and dropping and also if Ctrl is pressed
        # while dragging and dropping, CopyAction is performed instead
        self.setDefaultDropAction(Qt.MoveAction)
        # also this one
        self.setDragDropMode(QAbstractItemView.DragDrop)

        self.dragEnabled()
        self.acceptDrops()
        self.showDropIndicator()

        # self.myModel.dataChanged.connect(self.change)
        self.myModel.dataChanged.connect(lambda topLeftIndex, bottomRightIndex: self.update(topLeftIndex))

        # self.update(topLeftIndex)
        self.myModel.data_dropped_signal.connect(self.save_state)

        self.expandAll()

    def save_state(self):
        raise NotImplementedError("TODO")

    # https://stackoverflow.com/questions/50391050/how-to-remove-row-from-qtreeview-using-qabstractitemmodel
    def delete_selected_items(self):
        self.deleting = True

        # for any selected groups, we have to delete them first
        item_was_checked = False

        # --- First remove groups with all of their children ---

        sel_groups = []

        for item in self.myModel.iterate_selected_items(False, True):
            if isinstance(item, SpectrumItemGroup):
                sel_groups.append(item)

        for group in sel_groups:
            if group.check_state == Qt.Checked or group.check_state == Qt.PartiallyChecked:
                item_was_checked = True

            row = self.myModel.root.rowOfChild(group)
            # root_index = self.myModel.indexFromNode(self.myModel.root)

            # QModelIndex of root item is just plain QModelIndex()
            self.myModel.removeRows(row, 1, QModelIndex(), update_tristate=False)

        # --- Remove all remaining SpectrumItems ---

        # try to speed up thing for removing a lot af items, but still, it is kind of slow for > 1000 items...
        # but this is probably problem of Qt itself, because parent() and index() methods are called too many times
        # https://stackoverflow.com/questions/841096/slow-selection-in-qtreeview-why
        while len(self.selectedIndexes()) > 0:
            temp = []
            last_parent = None
            last_row = -1

            for item in self.myModel.iterate_selected_items(True, True):
                # item is SpectrumItem

                current_parent = item.parent

                if last_parent is None:
                    last_parent = current_parent

                current_row = current_parent.rowOfChild(item)

                if item.check_state == Qt.Checked:
                    item_was_checked = True

                if last_row == -1:
                    last_row = current_row

                if current_parent != last_parent or current_row - last_row > 1:
                    break
                else:
                    last_row = current_row
                    last_parent = current_parent
                    temp.append(item)

            if len(temp) == 0:
                continue

            parent = temp[0].parent
            row = parent.rowOfChild(temp[0])
            parent_index = self.myModel.index_from_node(parent)
            self.myModel.removeRows(row, len(temp), parent_index, update_tristate=False)

        self.deleting = False

        self.myModel.update_tristate()
        self.setup_info()

        # emit deleted event and if at least one of the deleted items were checked -> redraw spectra
        self.items_deleted_signal.emit(item_was_checked)
        self.save_state()

    def add_items_to_group(self, items, group=None, row_to_place=None, edit=True):
        """edit - if place cursor in a new group item to edit the name"""

        if not np.iterable(items):
            raise ValueError("Argument items must be iterable.")

        # add group to the end of root
        group_item = SpectrumItemGroup('', '', parent=self.myModel.root) if group is None else group
        self.myModel.insertRows(self.myModel.root.__len__(), 1, QModelIndex())

        # group_item_index = self.myModel.createIndex(self.myModel.root.__len__(), 0, group_item)

        def _move_items(parent, first_row, last_row, row_to_put):
            group_item_index = self.myModel.createIndex(self.myModel.root.__len__(), 0, group_item)
            source_parent = self.myModel.index_from_node(parent)

            self.myModel.beginMoveRows(source_parent, first_row, last_row, group_item_index, row_to_put)

            chunk = parent.children[first_row:last_row + 1]  # take the reference
            del parent.children[first_row:last_row + 1]  # delete them from from list

            for child in chunk:
                child.setParent(group_item)

            self.myModel.endMoveRows()

        last_parent = None
        first_row = -1
        last_row = -1
        i = 0

        for item in items:
            if row_to_place is None:
                row_to_place = item.parent.row() if item.is_in_group() else item.row()

            current_parent = item.parent

            if last_parent is None:
                last_parent = current_parent

            current_row = current_parent.rowOfChild(item)

            if first_row == -1:
                first_row = current_row

            if last_row == -1:
                last_row = current_row

            if current_parent != last_parent or current_row - last_row > 1:  # move chunk of items

                n_items = (last_row - first_row + 1)
                _move_items(last_parent, first_row, last_row, i - n_items)

                # current row will be changed because we removed n_items
                first_row = current_row if current_parent != last_parent else current_row - n_items
                last_row = first_row
            else:
                last_row = current_row

            last_parent = current_parent
            i += 1

        # process last chunk
        _move_items(last_parent, first_row, last_row, i - (last_row - first_row + 1))

        # move the group to correct place
        try:
            self.myModel.move_item(group_item, self.myModel.root, row_to_place)
        except:
            print("asdasd")

        index = self.myModel.index_from_node(group_item)

        self.expand(index)
        self.myModel.update_tristate()
        self.setup_info()

        # set edit mode of the group item
        if edit:
            self.edit(index)

        self.save_state()

    def add_selected_items_to_group(self):

        if len(self.selectedIndexes()) == 0:
            return

        iterator = self.myModel.iterate_selected_items(skip_groups=True,
                                                        skip_childs_in_selected_groups=False)
        self.add_items_to_group(iterator)

    def keyPressEvent(self, e):
        pass
        # if e.key() == 16777223:
        #     # pressed delete
        #     self.delete_selected_items()
        #
        # if e.key() == 65 and e.modifiers() == Qt.ControlModifier:
        #     # Ctrl + A
        #     self.selectAll()
        #
        # if e.key() == 86 and e.modifiers() == Qt.ControlModifier:
        #     # Ctrl + V
        #     self.add_selected_items_to_group()
        #
        # if e.key() == 66 and e.modifiers() == Qt.ControlModifier:
        #     # Ctrl + B
        #     self.create_group()
        #
        # if e.key() == 81 and e.modifiers() == Qt.ControlModifier:
        #     # Ctrl + Q
        #     self.check_selected_items()
        #
        # if e.key() == 87 and e.modifiers() == Qt.ControlModifier:
        #     # Ctrl + W
        #     self.uncheck_selected_items()

    def check_selected_items(self):
        if len(self.selectedIndexes()) == 0:
            return

        for item in self.myModel.iterate_selected_items(False, False):
            item.check_state = Qt.Checked

        self.myModel.update_tristate()
        self.myModel.dataChanged.emit(QModelIndex(), QModelIndex())
        self.myModel.checked_changed_signal.emit()

    def uncheck_selected_items(self):
        if len(self.selectedIndexes()) == 0:
            return

        for item in self.myModel.iterate_selected_items(False, False):
            item.check_state = Qt.Unchecked

        self.myModel.update_tristate()
        self.myModel.dataChanged.emit(QModelIndex(), QModelIndex())
        self.myModel.checked_changed_signal.emit()

    def uncheck_all(self):

        for item in self.myModel.iterate_items(ItemIterator.All):
            item.check_state = Qt.Unchecked

        self.myModel.dataChanged.emit(QModelIndex(), QModelIndex())
        self.myModel.checked_changed_signal.emit()

    def create_group(self):
        group_item = SpectrumItemGroup('', parent=self.myModel.root)
        self.myModel.insertRows(self.myModel.root.__len__(), 1, QModelIndex())
        self.edit(self.myModel.index_from_node(group_item))

    def sort_tree_view(self, sort_groups=False, ascending=True):

        if self.myModel.root.__len__() == 0:
            return

        self.myModel.layoutAboutToBeChanged.emit([])
        step = 1 if ascending else -1
        self.myModel.root.children = sorted(self.myModel.root.children, key=lambda child: child.name)[::step]
        self.myModel.layoutChanged.emit([])

        if sort_groups:
            self.myModel.layoutAboutToBeChanged.emit([])
            for group in self.myModel.iterate_items(ItemIterator.Groups):
                if group.__len__() == 0:
                    continue
                group.children = sorted(group.children, key=lambda child: child.name)[::step]
            self.myModel.layoutChanged.emit([])

        self.clearSelection()
        self.save_state()

    def sort_selected_group(self, ascending=True):
        self.sort_group(self.myModel.node_from_index(self.currentIndex()), ascending)

    def sort_group(self, group_item, ascending=True):
        # help from documentation for emitting signals:
        # http://doc.qt.io/qt-5/qabstractitemmodel.html#layoutAboutToBeChanged
        if group_item.__len__() == 0:
            return

        self.myModel.layoutAboutToBeChanged.emit([])
        step = 1 if ascending else -1
        group_item.children = sorted(group_item.children, key=lambda child: child.name)[::step]
        self.myModel.layoutChanged.emit([])
        self.clearSelection()

    def ungroup_selected_group(self):
        self.ungroup(self.myModel.node_from_index(self.currentIndex()))

    def ungroup(self, group_item):
        assert isinstance(group_item, SpectrumItemGroup)

        if group_item.__len__() == 0:
            return

        children = self.myModel.take_all_children(group_item)
        row = group_item.row()
        self.myModel.add_items(children, self.myModel.root, row + 1)

        del group_item

        self.myModel.removeRow(row, QModelIndex())

        self.save_state()


    # def change(self, topLeftIndex, bottomRightIndex):
    #     self.update(topLeftIndex)

    def clear(self):
        """Removes all items from the TreeView"""
        self.myModel.removeRows(0, self.top_level_items_count(), QModelIndex())

    def setup_info(self):
        """TODO --- rewrite more efficient"""

        for item in self.myModel.iterate_items(ItemIterator.All):

            if isinstance(item, SpectrumItemGroup):
                row = self.myModel.root.rowOfChild(item)
                it_len = item.__len__()
                item.info = "[{}] | {} item{}".format(row, it_len, '' if it_len == 1 else 's')
            else:

                # if item is None:
                #     continue
                if item.is_top_level():
                    row = self.myModel.root.rowOfChild(item)
                    item.info = "[{}] | {}; {} ({:.4g}, {:.4g})".format(row,
                                                                item.length(),
                                                                item.spacing(),
                                                                item.x.min(),
                                                                item.x.max())
                else:
                    parent = item.parent
                    group_idx = self.myModel.root.rowOfChild(parent)
                    item_idx = parent.rowOfChild(item)
                    item.info = "[{}][{}] | {}; {} ({:.4g}, {:.4g})".format(group_idx, item_idx,
                                                                    item.length(),
                                                                    item.spacing(),
                                                                    item.x.min(),
                                                                    item.x.max())

    def update_view(self):
        self.myModel.dataChanged.emit(QModelIndex(), QModelIndex())
        self.save_state()

    # def select_item(self, item):
    #     self.selecting = True
    #
    #     flags = QItemSelectionModel.Select
    #     selection = QItemSelection()
    #
    #     start_index = self.myModel.createIndex(item.row(), 0, item)
    #     end_index = self.myModel.createIndex(item.row(), 1, item)
    #
    #     selection.select(start_index, end_index)
    #     self.selectionModel().select(selection, flags)
    #
    #     self.selecting = False

    def selectionChanged(self, selected, deselected):

        super(TreeView, self).selectionChanged(selected, deselected)

        if self.selecting or self.deleting:
            return

        self.selecting = True

        # select all children if group was selected
        for item in self.myModel.iterate_selected_items(skip_groups=False, skip_childs_in_selected_groups=True):
            # help from https://stackoverflow.com/questions/47271494/set-selection-from-a-list-of-indexes-in-a-qtreeview
            if isinstance(item, SpectrumItemGroup):
                # mod = self.model()
                # columns = mod.columnCount() - 1

                length = item.__len__()

                if length == 0:
                    continue

                flags = QItemSelectionModel.Select
                selection = QItemSelection()

                start_index = self.myModel.createIndex(0, 0, item.children[0])
                end_index = self.myModel.createIndex(length - 1, 1, item.children[-1])

                selection.select(start_index, end_index)
                self.selectionModel().select(selection, flags)

        self.selecting = False

    def dragEnterEvent(self, e):
        if e.source():
            super(TreeView, self).dragEnterEvent(e)
            # print("dragEnterEvent - source")
        else:
            m = e.mimeData()
            if m.hasUrls():
                for url in m.urls():
                    if url.isLocalFile():
                        e.accept()
                return
            elif m.hasText():
                e.accept()
                return
            elif m.hasFormat("XML Spreadsheet"):
                e.accept()
                return
            elif m.hasFormat(PyMimeData.MIME_TYPE):
                super(TreeView, self).dragEnterEvent(e)
                # print("dragEnterEvent - from another app")
                # e.accept()
                return

            e.ignore()

    def dropEvent(self, e):

        if e.source():
            super(TreeView, self).dropEvent(e)
        else:
            m = e.mimeData()

            if m.hasFormat(PyMimeData.MIME_TYPE):
                super(TreeView, self).dropEvent(e)
                return

            if m.hasUrls():
                filepaths = []
                for url in m.urls():
                    # print(url.toString(), url.path())
                    if url.isLocalFile():
                        # e.accept()
                        if platform == "win32":
                            path = url.path()[1:]  # and  .replace('/', '\\')
                        else:
                            path = url.path()
                        filepaths.append(path)
                self.import_files(filepaths)
                return

            if m.hasFormat("XML Spreadsheet"):
                self.parse_XML_Spreadsheet(m.data("XML Spreadsheet").data())
                return
            e.ignore()

    @abstractmethod
    def import_files(self, filepaths):
        raise NotImplementedError

    @abstractmethod
    def parse_XML_Spreadsheet(self, byte_data):
        raise NotImplementedError

    def expanded(self, index):
        for column in range(self.model().columnCount(QModelIndex())):
            self.resizeColumnToContents(column)

    def top_level_items_count(self):
        return self.myModel.root.__len__()


if __name__ == "__main__":
    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    from PyQt5.QtWidgets import QApplication

    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    app = QApplication(sys.argv)
    MainWindow = TreeView()
    # ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    MainWindow.show()
    # sys.exit(app.exec_())

    cProfile.run('app.exec_()', 'profdata')
    p = pstats.Stats('profdata')
    p.sort_stats('time').print_stats()
