from PyQt5.QtCore import Qt, QItemSelectionModel, QItemSelection, pyqtSignal, QModelIndex
from PyQt5.QtWidgets import QApplication, QMessageBox, QMenu, QAction
from PyQt5.QtGui import QCursor, QColor
import numpy as np
import os


from spectramanipulator.spectrum import Spectrum, SpectrumList

from spectramanipulator.dialogs.int_int_inputdialog import IntIntInputDialog
from spectramanipulator.dialogs.interpolate_dialog import InterpolateDialog
from spectramanipulator.dialogs.rename_dialog import RenameDialog
from spectramanipulator.dialogs.fitwidget import FitWidget
from spectramanipulator.dialogs.stylewidget import StyleWidget
# from dialogs.rangedialog import RangeDialog
from spectramanipulator.dialogs.rangewidget import RangeWidget
from spectramanipulator.dialogs.export_spectra_as import ExportSpectraAsDialog

from .settings.settings import Settings
from .settings.structure import get_delimiter_from_idx
from spectramanipulator.logger import Logger
from spectramanipulator.utils.rename import rename

from spectramanipulator.treeview.item import SpectrumItemGroup, SpectrumItem, GenericItem
from spectramanipulator.treeview.model import TreeView, ItemIterator

from spectramanipulator.console import Console

from spectramanipulator.parsers import parse_XML_Spreadsheet
from spectramanipulator.dataloader import parse_text, parse_files
from spectramanipulator.exporter import list_to_string, list_to_files
# from spectramanipulator.plotwidget import PlotWidget


# TODO-->> rewrite
def get_hierarchic_list(items_iterator):
    """ get a hierarchic structure of selected items
    spectra in groups are appended in list of spectrum objects
    spectra not in groups are appended as spectrum objects
    """

    sp_list = []
    temp_list = []
    last_grp_item = None

    for item in items_iterator:
        if isinstance(item, SpectrumItemGroup):
            continue

        item.name = item.name
        if item.is_in_group():
            curr_grp_item = item.parent
            item.group_name = curr_grp_item.name
            if last_grp_item != curr_grp_item:
                last_grp_item = curr_grp_item
                if len(temp_list) != 0:
                    sp_list.append(temp_list)
                    temp_list = []
                temp_list.append(item)
            else:
                temp_list.append(item)
        else:
            if len(temp_list) > 0:
                sp_list.append(temp_list)
                temp_list = []
            item.group_name = None
            sp_list.append(item)

    if len(temp_list) > 0:
        sp_list.append(temp_list)

    return sp_list


class TreeWidget(TreeView):
    redraw_spectra = pyqtSignal()
    state_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(TreeWidget, self).__init__(parent)

        self.main_widget = self.parent().parent()

        self.header().setStretchLastSection(True)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu)

        self.items_deleted_signal.connect(self.items_deleted)
        self.myModel.item_edited_signal.connect(self.item_edited)
        self.myModel.checked_changed_signal.connect(self.check_changed)
        self.myModel.data_dropped_signal.connect(self.data_dropped)
        self.myModel.all_unchecked_signal.connect(lambda: self.redraw_spectra.emit())
        self.myModel.data_modified_signal.connect(self.data_modified)
        self.myModel.info_modified_signal.connect(self.info_modified)
        self.myModel.items_ungrouped_signal.connect(lambda: self.redraw_spectra.emit())

        self.sett = Settings()

    def items_deleted(self, item_was_checked):
        if item_was_checked:
            self.redraw_spectra.emit()

    def item_edited(self, item_is_checked):
        self.state_changed.emit()
        if item_is_checked:
            self.redraw_spectra.emit()

    def data_modified(self, items):
        self.main_widget.update_items_data(items)

    def info_modified(self, items):
        self.setup_info()
        self.update_view()

    def check_changed(self, items, checked):
        # self.redraw_spectra.emit()
        self.main_widget.redraw_items(items, not checked)

    def data_dropped(self):
        self.redraw_spectra.emit()

    def save_state(self):
        super(TreeWidget, self).save_state()
        self.state_changed.emit()

        # if self.top_level_items_count() == 0:
        #     self.all_spectra_list = []
        #     return

        # self.all_spectra_list = self.get_hierarchic_list(
        #     self.myModel.iterate_items(ItemIterator.NoChildren))

        # Console.push_variables({'item': self.all_spectra_list})
        # Console.push_variables({'item': self.myModel.root})

    def export_selected_items_as(self):

        if len(self.selectedIndexes()) == 0:
            return

        def accepted():
            path, ext, delimiter, decimal_sep = dialog.result

            sp_list = get_hierarchic_list(
                self.myModel.iterate_selected_items(skip_groups=True,
                                                    skip_childs_in_selected_groups=False))

            try:
                list_to_files(sp_list, path, ext,
                              include_group_name=self.sett['/Public settings/Export/Files/Include group name'],
                              include_header=self.sett['/Public settings/Export/Files/Include header'],
                              delimiter=delimiter,
                              decimal_sep=decimal_sep,
                              x_data_name=self.sett['/Public settings/Plotting/Graph/X axis label'])

            except Exception as ex:
                QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)

            Logger.message(f"Data were saved to {path}")

        dialog = ExportSpectraAsDialog(accepted, parent=self)
        dialog.show()

    def copy_selected_items_to_clipboard(self):

        sp_list = get_hierarchic_list(self.myModel.iterate_selected_items(skip_groups=True,
                                                                          skip_childs_in_selected_groups=False))

        if len(sp_list) == 0:
            return

        Logger.status_message("Copying selected items to clipboard...")
        try:

            delimiter = get_delimiter_from_idx(self.sett['/Public settings/Export/Clipboard/Delimiter'])

            output = list_to_string(sp_list, include_group_name=self.sett['/Public settings/Export/Clipboard/Include group name'],
                                    include_header=self.sett['/Public settings/Export/Clipboard/Include header'],
                                    delimiter=delimiter,
                                    decimal_sep=self.sett['/Public settings/Export/Clipboard/Decimal separator'],
                                    x_data_name=self.sett['/Public settings/Plotting/Graph/X axis label'])
            cb = QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(output, mode=cb.Clipboard)

        except Exception as ex:
            Logger.message(ex.__str__())
            return

        Logger.status_message("Done")

    def paste_from_clipboard(self):

        m = QApplication.clipboard().mimeData()

        if m is not None and m.hasFormat("XML Spreadsheet") and not self.sett['/Public settings/Import/Parser/Clipboard/Import as text from Excel']:
            self.parse_XML_Spreadsheet(m.data("XML Spreadsheet").data())
            return

        cb = QApplication.clipboard()
        text_data = cb.text(mode=cb.Clipboard)
        spectra = parse_text(text_data)

        self.import_spectra(spectra)

    # --------- Operation Functions --------

    def normalize(self):

        if len(self.selectedIndexes()) == 0:
            return

        def accepted():
            x0, x1 = rng_dialog.returned_range

            try:
                items = []
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.normalize_no_update(x0, x1)
                    items.append(item)

                self.data_modified(items)
                self.state_changed.emit()

            except ValueError as ex:
                Logger.message(ex.__str__())
                QMessageBox.warning(self, 'Warning', ex.__str__(), QMessageBox.Ok)
                return

            Logger.status_message("Done")

        rng_dialog = RangeWidget(self.main_widget.var_widget, accepted, title="Normalize",
                                 label_text="Set x range values. Maximum y value is find in this range and y values "
                                            "are divided by this maximum. For normalizing to specific value, set x0 "
                                            "equal to x1:",
                                 parent=self)

    def cut(self):

        if len(self.selectedIndexes()) == 0:
            return

        def accepted():
            x0, x1 = rng_dialog.returned_range

            # Logger.status_message("Cutting the spectra...")

            try:
                items = []
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.cut_no_update(x0, x1)
                    items.append(item)

                self.data_modified(items)
                self.info_modified(items)
                self.state_changed.emit()

            except ValueError as ex:
                Logger.message(ex.__str__())
                QMessageBox.warning(self, 'Warning', ex.__str__(), QMessageBox.Ok)
                return

            Logger.status_message("Done")

        rng_dialog = RangeWidget(self.main_widget.var_widget, accepted, title="Cut",
                                 label_text="Set x range values. Values outside this range will"
                                            " be deleted from the spectrum/spectra:",
                                 parent=self)

    def extend_by_zeros(self):

        if len(self.selectedIndexes()) == 0:
            return

        def accepted():
            x0, x1 = rng_dialog.returned_range

            try:
                items = []
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.extend_by_value_no_update(x0, x1)
                    items.append(item)

                self.data_modified(items)
                self.info_modified(items)
                self.state_changed.emit()

            except ValueError as ex:
                Logger.message(ex.__str__())
                QMessageBox.warning(self, 'Warning', ex.__str__(), QMessageBox.Ok)
                return

            Logger.status_message("Done")

        rng_dialog = RangeWidget(self.main_widget.var_widget, accepted, title="Extend by zeros",
                                 label_text="Set x range values. Values outside the range of spectrum "
                                            "will be set up to zero. Spacing is calculated as average "
                                            "of the spectrum spacing (spacing will be the same for "
                                            "evenly spaced spectrum:",
                                 parent=self)

    def baseline_correct(self):

        if len(self.selectedIndexes()) == 0:
            return

        def accepted():
            # Logger.status_message("Baseline correcting...")
            x0, x1 = rng_dialog.returned_range
            try:
                items = []
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.baseline_correct_no_update(x0, x1)
                    items.append(item)

                self.data_modified(items)
                self.state_changed.emit()
            except ValueError as ex:
                Logger.message(ex.__str__())
                QMessageBox.warning(self, 'Warning', ex.__str__(), QMessageBox.Ok)
                return

            Logger.status_message("Done")

        rng_dialog = RangeWidget(self.main_widget.var_widget, accepted, title="Baseline correction",
                                 label_text="Set x range values. Y values of this range will "
                                            "be averaged and subtracted from all points:",
                                 parent=self)

    def interpolate(self):

        if len(self.selectedIndexes()) == 0:
            return

        def accepted():
            # Logger.status_message("Baseline correcting...")
            try:
                items = []
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.interpolate_no_update(interp_dialog.spacing, interp_dialog.selected_kind)
                    items.append(item)

                self.data_modified(items)
                self.info_modified(items)
                self.state_changed.emit()
            except ValueError as ex:
                Logger.message(ex.__str__())
                QMessageBox.warning(self, 'Warning', ex.__str__(), QMessageBox.Ok)
                return

            Logger.status_message("Done")

        interp_dialog = InterpolateDialog(accepted, parent=self)
        interp_dialog.show()

    def select_every_nth_item(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        def accepted():
            try:
                shift, n = intintinput_dialog.sbOffset.value(), intintinput_dialog.sbn.value()
                shift = shift % n
                i = 0

                self.selecting = True

                flags = QItemSelectionModel.Select
                selection = QItemSelection()

                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False,
                                                                clear_selection=True):
                    if i % n == shift:
                        start_index = self.myModel.createIndex(item.row(), 0, item)
                        end_index = self.myModel.createIndex(item.row(), 1, item)

                        selection.select(start_index, end_index)

                    i += 1
                self.selectionModel().select(selection, flags)

            except Exception as ex:
                Logger.message(ex.__str__())
                QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)
            finally:
                self.selecting = False

        intintinput_dialog = IntIntInputDialog(accepted, 2, 0, n_min=1, offset_min=0,
                                               title="Select every n-th item",
                                               label="Set the n value and shift value. Group items will be skipped:")
        intintinput_dialog.show()

    def fit_curve(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        selected_node = self.myModel.node_from_index(self.selectedIndexes()[0])
        # if isinstance(selected_node, SpectrumItemGroup):  # TODO>>>>
        #     return

        def accepted():
            self.import_spectra([fit_dialog.fits, fit_dialog.residuals])
            self.state_changed.emit()

        fit_dialog = FitWidget(self.main_widget.var_widget, accepted, selected_node, parent=self)
        Console.push_variables({'fw': fit_dialog})

    def set_style(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        selected_node = self.myModel.node_from_index(self.selectedIndexes()[0])
        if isinstance(selected_node, SpectrumItemGroup):
            selected_node = selected_node[0]  # select spectrum

        def accepted():
            items = []
            for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                            skip_childs_in_selected_groups=False):
                color = None
                if not style_widget.cbColor.isChecked() and style_widget.color is not None:
                    color = style_widget.color.name(QColor.HexRgb)  # in rgb format

                line_alpha = int(style_widget.sbAlpha.value())

                line_width = None if style_widget.cbLineWidth.isChecked() else float(
                    style_widget.dsbLineWidth.value())
                line_type = None if style_widget.cbLineType.isChecked() else \
                    style_widget.line_types[style_widget.combLineType.currentIndex()]['index']

                symbol = style_widget.symbol_types[style_widget.combSymbol.currentIndex()]['sym']

                sym_brush_color = None
                if not style_widget.cbSymBrushDefault.isChecked() and style_widget.sym_brush_color is not None:
                    # style_widget.sym_brush_color.setAlpha(style_widget.sbSymBrushAlpha.value())
                    sym_brush_color = style_widget.sym_brush_color.name(QColor.HexRgb)  # in rgb format

                sym_fill_color = None
                if not style_widget.cbSymFillDefault.isChecked() and style_widget.sym_fill_color is not None:
                    # style_widget.sym_fill_color.setAlpha(style_widget.sbSymFillAlpha.value())
                    sym_fill_color = style_widget.sym_fill_color.name(QColor.HexRgb)  # in rgb format

                sym_brush_alpha = int(style_widget.sbSymBrushAlpha.value())
                sym_fill_alpha = int(style_widget.sbSymFillAlpha.value())
                symbol_size = float(style_widget.dsbSymSize.value())

                plot_legend = style_widget.cbPlotLegend.isChecked()

                if not hasattr(item, 'plot_legend'):
                    setattr(item, 'plot_legend', plot_legend)

                item.color = color
                item.line_width = line_width
                item.line_type = line_type
                item.plot_legend = plot_legend

                setattr(item, 'symbol', symbol)
                setattr(item, 'symbol_brush', sym_brush_color)
                setattr(item, 'symbol_fill', sym_fill_color)
                setattr(item, 'symbol_size', symbol_size)

                setattr(item, 'line_alpha', line_alpha)
                setattr(item, 'sym_brush_alpha', sym_brush_alpha)
                setattr(item, 'sym_fill_alpha', sym_fill_alpha)

                items.append(item)

            self.check_changed(items, True)
            self.state_changed.emit()

        style_widget = StyleWidget(self.main_widget.var_widget, accepted, selected_node, parent=self)

    def rename(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        def accepted():

            if rename_dialog.cbTakeNamesFromList.isChecked():
                import csv
                splitted_list = csv.reader([rename_dialog.leList.text()], doublequote=True, skipinitialspace=True,
                                           delimiter=',').__next__()

                if len(splitted_list) == 0:
                    return
            expression, offset, c_mult_factor = rename_dialog.leExpression.text(), rename_dialog.sbOffset.value(), rename_dialog.leCounterMulFactor.text()

            try:
                i = 0
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    name = item.name
                    if not rename_dialog.cbTakeNamesFromList.isChecked():
                        name = rename(expression, name, float(offset) * float(c_mult_factor))
                        offset += 1
                    else:
                        try:
                            name = splitted_list[i].strip()
                            i += 1
                        except IndexError:
                            pass
                    item.name = name

                self.redraw_spectra.emit()
                self.update_view()
                self.state_changed.emit()

                self.sett['/Private settings/Last rename expression'] = expression
                self.sett['/Private settings/Last rename name taken from list'] = rename_dialog.cbTakeNamesFromList.isChecked()
                self.sett.save()
            except Exception as ex:
                Logger.message(ex.__str__())
                QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)

        rename_dialog = RenameDialog(accepted, self.sett['/Private settings/Last rename expression'], 0, 1,
                                     last_rename_take_name_from_list=self.sett[
                                         '/Private settings/Last rename name taken from list'])
        rename_dialog.show()

    def context_menu(self, pos):
        """Creates a context menu in a TreeWidget."""

        # pos is local position on QTreeWidget
        # cursor.pos() is position on screen

        item = self.myModel.node_from_index(self.indexAt(pos))

        # print(pos, item.text(0) if item is not None else "None")

        menu = QMenu()

        sel_it_menu = QMenu("With Selected Items")

        check_selected_items = QAction("Check Items (Ctrl + Q)", self)
        # check_selected_items.setShortcut(QKeySequence(Qt.Key_Control, Qt.Key_D))

        sel_it_menu.addAction(check_selected_items)
        check_selected_items.triggered.connect(self.check_selected_items)

        uncheck_selected_items = sel_it_menu.addAction("Uncheck Items (Ctrl + W)")
        uncheck_selected_items.triggered.connect(self.uncheck_selected_items)

        select_every_nth = sel_it_menu.addAction("Select Every n-th Item (Ctrl + D)")
        select_every_nth.triggered.connect(self.select_every_nth_item)

        rename = sel_it_menu.addAction("Rename Items (Ctrl + R)")
        rename.triggered.connect(self.rename)

        sel_it_menu.addSeparator()

        cut = sel_it_menu.addAction("Cut (Ctrl + T)")
        cut.triggered.connect(self.cut)

        baseline_correct = sel_it_menu.addAction("Baseline Correct (Ctrl + B)")
        baseline_correct.triggered.connect(self.baseline_correct)

        normalize = sel_it_menu.addAction("Normalize (Ctrl + N)")
        normalize.triggered.connect(self.normalize)

        extend_by_zeros = sel_it_menu.addAction("Extend By Zeros")
        extend_by_zeros.triggered.connect(self.extend_by_zeros)

        interpolate = sel_it_menu.addAction("Interpolate")
        interpolate.triggered.connect(self.interpolate)

        fit_curve = sel_it_menu.addAction("Fit Curve")
        fit_curve.triggered.connect(self.fit_curve)

        sel_it_menu.addSeparator()

        set_style = sel_it_menu.addAction("Set Style")
        set_style.triggered.connect(self.set_style)

        copy_to_clipboard = sel_it_menu.addAction("Copy Items To Clipboard (Ctrl + C)")
        copy_to_clipboard.triggered.connect(self.copy_selected_items_to_clipboard)

        export = sel_it_menu.addAction("Export Items As")
        export.triggered.connect(self.export_selected_items_as)

        # sort group menu

        sort_group_menu = QMenu("Sort Group")

        sort_group_ascending = QAction("Ascending", self)
        sort_group_descending = QAction("Descending", self)

        sort_group_menu.addAction(sort_group_ascending)
        sort_group_menu.addAction(sort_group_descending)
        sort_group_ascending.triggered.connect(lambda: self.sort_selected_group(ascending=True))
        sort_group_descending.triggered.connect(lambda: self.sort_selected_group(ascending=False))

        # sort top level items menu

        sort_top_level_items_menu = QMenu("Sort Top-level Items")

        sort_tli_ascending = QAction("Ascending", self)
        sort_tli_descending = QAction("Descending", self)

        sort_top_level_items_menu.addAction(sort_tli_ascending)
        sort_top_level_items_menu.addAction(sort_tli_descending)
        sort_tli_ascending.triggered.connect(lambda: self.sort_tree_view(ascending=True))
        sort_tli_descending.triggered.connect(lambda: self.sort_tree_view(ascending=False))

        # sort all items also withing all groups

        sort_all_menu = QMenu("Sort All Items and Groups")

        sort_all_ascending = QAction("Ascending", self)
        sort_all_descending = QAction("Descending", self)

        sort_all_menu.addAction(sort_all_ascending)
        sort_all_menu.addAction(sort_all_descending)
        sort_all_ascending.triggered.connect(lambda: self.sort_tree_view(sort_groups=True, ascending=True))
        sort_all_descending.triggered.connect(lambda: self.sort_tree_view(sort_groups=True, ascending=False))

        # clicked into blank
        if item is self.myModel.root:
            create_group = menu.addAction("Create a New Group")
            create_group.triggered.connect(self.create_group)

            uncheck_all = menu.addAction("Uncheck All Items")
            uncheck_all.triggered.connect(self.uncheck_all)

            menu.addSeparator()

            menu.addMenu(sort_top_level_items_menu)
            menu.addMenu(sort_all_menu)
        else:
            # Clicked on item
            if isinstance(item, SpectrumItem):
                add_to_group = menu.addAction("Move Items to New Group")
                add_to_group.triggered.connect(lambda: self.add_selected_items_to_group(copy=False))

                copy_to_group = menu.addAction("Copy Items to New Group")
                copy_to_group.triggered.connect(lambda: self.add_selected_items_to_group(copy=True))

                uncheck_all = menu.addAction("Uncheck All Items")
                uncheck_all.triggered.connect(self.uncheck_all)

                menu.addMenu(sel_it_menu)

                menu.addSeparator()

            # Clicked on group
            else:
                ungroup = menu.addAction("Ungroup")
                ungroup.triggered.connect(self.ungroup_selected_group)

                uncheck_all = menu.addAction("Uncheck All Items")
                uncheck_all.triggered.connect(self.uncheck_all)

                menu.addMenu(sel_it_menu)

                menu.addSeparator()

                menu.addMenu(sort_group_menu)

            menu.addMenu(sort_top_level_items_menu)
            menu.addMenu(sort_all_menu)

            menu.addSeparator()

            delete_items = menu.addAction("Remove")
            delete_items.triggered.connect(self.delete_selected_items)

        menu.addSeparator()

        paste_from_clipboard = menu.addAction("Paste Data From Clipboard (Ctrl + V)")
        paste_from_clipboard.triggered.connect(self.paste_from_clipboard)

        cursor = QCursor()

        menu.exec_(cursor.pos())

    def keyPressEvent(self, e):
        # e is QKeyEvent object
        #
        # print(e.key())
        # return

        if e.key() == 16777223:  # pressed delete
            self.delete_selected_items()

        if e.key() == 67 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + C
            self.copy_selected_items_to_clipboard()

        if e.key() == 86 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + V
            self.paste_from_clipboard()

        if e.key() == 65 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + A
            self.selectAll()

        if e.key() == 66 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + B
            self.baseline_correct()

        if e.key() == 82 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + R
            self.rename()

        if e.key() == 78 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + N
            self.normalize()

        if e.key() == 81 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + Q
            self.check_selected_items()

        if e.key() == 87 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + W
            self.uncheck_selected_items()

        if e.key() == 84 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + T
            self.cut()

        if e.key() == 68 and e.modifiers() == Qt.ControlModifier:
            # Ctrl + D
            self.select_every_nth_item()

    def import_project(self, generic_item):
        if generic_item is None:
            return

        if not hasattr(generic_item, 'children'):
            raise ValueError("Argument 'generic_item' must have an attribute 'children'.")

        num_items = len(generic_item.children)

        if num_items == 0:
            return

        self.myModel.root.children += generic_item.children

        for child in self.myModel.root.children:  # set the parent to be the current root
            child.parent = self.myModel.root

        del generic_item  # destroy the original object

        if num_items > 0:
            # update names and view
            self.myModel.insertRows(self.myModel.root.__len__(), num_items, QModelIndex())
            self.setup_info()
            self.save_state()

        self.redraw_spectra.emit()

    def add_to_list(self, items):
        """
        Copies all spectra and import them to the treewidget
        :param spectra: input parameter can be single spectrum or
        spectrumlist object or simple list of spectrast of spectra.
        """

        if not isinstance(items, list):
            items = [items.__copy__()]

        self.import_spectra(items)

    def import_spectra(self, spectra):
        if spectra is None:
            return

        if not isinstance(spectra, list):
            raise ValueError("Argument spectra must be type of list.")

        if len(spectra) == 0:
            return

        add_rows_count = 0

        for node in spectra:
            if isinstance(node, SpectrumItem):
                # child = SpectrumItem(node, node.name, '', parent=self.myModel.root)
                node.setParent(self.myModel.root)
                add_rows_count += 1
            elif isinstance(node, Spectrum):
                SpectrumItem.from_spectrum(node, parent=self.myModel.root)
                add_rows_count += 1
            if isinstance(node, list):  # for backward compatibility
                group_item = SpectrumItemGroup(name=node[0].group_name, info='', parent=self.myModel.root)
            elif isinstance(node, SpectrumList):  # list of spectra
                group_item = SpectrumItemGroup(name=node.name, info='', parent=self.myModel.root)
            else:
                continue

            add_rows_count += 1

            for sp in node:
                if isinstance(sp, SpectrumItem):
                    sp.setParent(group_item)
                else:
                    SpectrumItem.from_spectrum(sp, parent=group_item)

        if add_rows_count > 0:
            # update view and names
            self.myModel.insertRows(self.myModel.root.__len__(), add_rows_count, QModelIndex())
            self.setup_info()
            self.save_state()

    def parse_XML_Spreadsheet(self, byte_data):

        Logger.status_message("Parsing XML Spreadsheet from clipboard Mime data...")
        try:
            # get bytes from QMimeData, remove the last character and convert to string
            xml_data = byte_data[:-1].decode("utf-8")

            spectra = parse_XML_Spreadsheet(xml_data)
            self.import_spectra(spectra)
        except Exception as ex:
            Logger.message("Error reading XML Spreadsheet from clipboard Mime data.\n{}".format(ex.__str__()))
            return

        Logger.status_message("Done")

    def import_files(self, filepaths):

        # if the dropped file is a project file, load it
        if filepaths[0].lower().endswith(Settings.PROJECT_EXTENSION):
            self.main_widget.open_project(filepaths[0], open_dialog=False)
            return

        spectra, _ = parse_files(filepaths)
        self.import_spectra(spectra)

    def load_kinetics(self, dir_name, spectra_dir_name='spectra', times_fname='times.txt', blank_spectrum='blank.dx', dt=None,
                      b_corr=None, cut=None, corr_to_zero_time=True):
        """Given a directory name that contains folders of individual experiments, it loads all kinetics.
           each experiment folder must contain folder spectra (or defined in spectra_dir_name arg.)
           if blank is given, it will be subtracted from all spectra, times.txt will contain
           times for all spectra, optional baseline correction and cut can be done.

        Folder structure:
            [dir_name]
                [exp1_dir]
                    [spectra]
                        01.dx (or .csv or .txt)
                        02.dx
                        ...
                    times.txt (optional)
                    blank.dx (optional)
                [exp2_dir]
                    ...
                ...
        """

        if not os.path.isdir(dir_name):
            raise ValueError(f'{dir_name}  does not exist!')

        for item in os.listdir(dir_name):
            path = os.path.join(dir_name, item)
            if not os.path.isdir(path):
                continue

            self.load_kinetic(path, spectra_dir_name=spectra_dir_name, times_fname=times_fname, blank_spectrum=blank_spectrum,
                              dt=dt, b_corr=b_corr, cut=cut, corr_to_zero_time=corr_to_zero_time)

    def load_kinetic(self, dir_name, spectra_dir_name='spectra', times_fname='times.txt', blank_spectrum='blank.dx', dt=None,
                     b_corr=None, cut=None, corr_to_zero_time=True):
        """Given a directory name, it loads all spectra in dir named "spectra" - func. arg.,
        if blank is given, it will be subtracted from all spectra, times.txt will contain
        times for all spectra, optional baseline correction and cut can be done.

        Folder structure:
            [dir_name]
                [spectra]
                    01.dx
                    02.dx
                    ...
                times.txt (optional)
                blank.dx (optional)
        """

        root = self.myModel.root  # item in IPython console

        if not os.path.isdir(dir_name):
            raise ValueError(f'{dir_name}  does not exist!')

        spectra_path = os.path.join(dir_name, spectra_dir_name)

        if not os.path.isdir(spectra_path):
            raise ValueError(f'{spectra_dir_name}  does not exist in {dir_name}!')

        spectras = [os.path.join(spectra_path, filename) for filename in os.listdir(spectra_path)]

        n_items_before = root.__len__()
        self.import_files(spectras)
        n_spectra = root.__len__() - n_items_before

        self.add_items_to_group(root[n_items_before:], edit=False)  # add loaded spectra to group
        root[n_items_before].name = f'raw [{os.path.split(dir_name)[1]}]'  # set name of a group

        times = np.asarray([dt * i for i in range(n_spectra)]) if dt is not None else None
        # idx_add = 0
        group_idx = n_items_before
        blank_used = False

        # load explicit times
        times_fpath = os.path.join(dir_name, times_fname)
        if times is None and os.path.isfile(times_fpath):
            self.import_files(times_fpath)
            # idx_add += 1
            times = root[-1].data[:, 0].copy()
            if corr_to_zero_time:
                times -= times[0]

        if times is not None:
            root[group_idx].set_names(times)

        # load blank spectrum if available
        blank_fpath = os.path.join(dir_name, blank_spectrum)
        if os.path.isfile(blank_fpath):
            last_idx = root.__len__() - 1
            self.import_files(blank_fpath)
            self.add_to_list(root[group_idx] - root[last_idx + 1])
            if times is not None:
                root[-1].set_names(times)
            blank_used = True

        corr_idx = -1 if blank_used else group_idx

        if b_corr is not None:
            root[corr_idx].baseline_correct(*b_corr)
            root[corr_idx].name += 'bcorr'
        if cut is not None:
            root[corr_idx].cut(*cut)
            root[corr_idx].name += 'cut'



