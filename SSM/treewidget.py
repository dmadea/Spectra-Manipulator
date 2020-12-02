from PyQt5.QtCore import Qt, QItemSelectionModel, QItemSelection, pyqtSignal, QModelIndex
from PyQt5.QtWidgets import QApplication, QMessageBox, QMenu, QAction
from PyQt5.QtGui import QCursor, QColor


from SSM import Spectrum  # , SpectrumList

from SSM.dialogs.int_int_inputdialog import IntIntInputDialog
from SSM.dialogs.interpolate_dialog import InterpolateDialog
from SSM.dialogs.rename_dialog import RenameDialog
from SSM.dialogs.fitwidget import FitWidget
from SSM.dialogs.stylewidget import StyleWidget
# from dialogs.rangedialog import RangeDialog
from SSM.dialogs.rangewidget import RangeWidget
from SSM.dialogs.export_spectra_as import ExportSpectraAsDialog

from SSM import Settings, Logger
from SSM.utils.smart_rename import smart_rename

from SSM.treeview.item import SpectrumItemGroup, SpectrumItem
from SSM.treeview.model import TreeView, ItemIterator

# from console import Console

from SSM.parsers import parse_XML_Spreadsheet
from SSM.dataparser import parse_text, parse_files
from SSM.exporter import list_to_string, list_to_files


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

    # all_spectra_list =

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

    def items_deleted(self, item_was_checked):
        if item_was_checked:
            self.redraw_spectra.emit()

    def item_edited(self, item_is_checked):
        self.state_changed.emit()
        if item_is_checked:
            self.redraw_spectra.emit()

    def check_changed(self):
        self.redraw_spectra.emit()

    def data_dropped(self):
        self.redraw_spectra.emit()

    def save_state(self):

        self.state_changed.emit()

        # if self.top_level_items_count() == 0:
        #     self.all_spectra_list = []
        #     return

        # self.all_spectra_list = self.get_hierarchic_list(
        #     self.myModel.iterate_items(ItemIterator.NoChildren))

        # Console.push_variables({'item': self.all_spectra_list})
        # Console.push_variables({'item': self.myModel.root})

    def export_selected_items_as(self):

        if ExportSpectraAsDialog.is_opened:
            ExportSpectraAsDialog.get_instance().activateWindow()
            ExportSpectraAsDialog.get_instance().setFocus()
            return

        if len(self.selectedIndexes()) == 0:
            return

        dialog = ExportSpectraAsDialog()

        if not dialog.accepted:
            return

        path, ext, delimiter, decimal_sep = dialog.result

        sp_list = get_hierarchic_list(
            self.myModel.iterate_selected_items(skip_groups=True,
                                                skip_childs_in_selected_groups=False))

        try:
            list_to_files(sp_list, path, ext, include_group_name=Settings.files_exp_include_group_name,
                                   include_header=Settings.files_exp_include_header,
                                   delimiter=delimiter,
                                   decimal_sep=decimal_sep,
                                   x_data_name=Settings.bottom_axis_label)

        except Exception as ex:
            QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)

        Logger.message(f"Data were saved to {path}")

    def copy_selected_items_to_clipboard(self):

        sp_list = get_hierarchic_list(self.myModel.iterate_selected_items(skip_groups=True,
                                                                          skip_childs_in_selected_groups=False))

        if len(sp_list) == 0:
            return

        Logger.status_message("Copying selected items to clipboard...")
        try:

            output = list_to_string(sp_list, include_group_name=Settings.clip_exp_include_group_name,
                                             include_header=Settings.clip_exp_include_header,
                                             delimiter=Settings.clip_exp_delimiter,
                                             decimal_sep=Settings.clip_exp_decimal_sep,
                                             x_data_name=Settings.bottom_axis_label)
            cb = QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(output, mode=cb.Clipboard)

        except Exception as ex:
            Logger.message(ex.__str__())
            return

        Logger.status_message("Done")

    def paste_from_clipboard(self):

        m = QApplication.clipboard().mimeData()

        if m is not None and m.hasFormat("XML Spreadsheet") and not Settings.excel_imp_as_text:
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
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.normalize(x0, x1, False)
                self.redraw_spectra.emit()
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
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.cut(x0, x1, False)

                self.redraw_spectra.emit()
                self.setup_info()
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
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.extend_by_zeros(x0, x1, False)

                self.redraw_spectra.emit()
                self.setup_info()
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
                for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                                skip_childs_in_selected_groups=False):
                    item.baseline_correct(x0, x1, False)

                self.redraw_spectra.emit()
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

        if InterpolateDialog.is_opened:
            InterpolateDialog.get_instance().activateWindow()
            InterpolateDialog.get_instance().setFocus()
            return

        interp_dialog = InterpolateDialog(self)

        if not interp_dialog.accepted:
            return

        spacing, kind = interp_dialog.spacing, interp_dialog.selected_kind

        try:
            for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                            skip_childs_in_selected_groups=False):
                item.interpolate(spacing, kind, False)

            self.redraw_spectra.emit()
            self.setup_info()
            self.state_changed.emit()
        except ValueError as ex:
            Logger.message(ex.__str__())
            QMessageBox.warning(self, 'Warning', ex.__str__(), QMessageBox.Ok)
            return

        Logger.status_message("Done")

    def select_every_nth_item(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        if IntIntInputDialog.is_opened:
            IntIntInputDialog.get_instance().activateWindow()
            IntIntInputDialog.get_instance().setFocus()
            return

        n, shift = 2, 0

        intintinput_dialog = IntIntInputDialog(n, shift, n_min=1, offset_min=0,
                                               title="Select every n-th item",
                                               label="Set the n value and shift value. Group items will be skipped:")
        if not intintinput_dialog.accepted:
            return
        n, shift = intintinput_dialog.returned_range

        try:
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
            # self.selecting = False

        except Exception as ex:
            Logger.message(ex.__str__())
            QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)
        finally:
            self.selecting = False

    def fit_curve(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        selected_node = self.myModel.node_from_index(self.selectedIndexes()[0])
        if isinstance(selected_node, SpectrumItemGroup):  # TODO>>>>
            return

        def accepted():
            self.import_spectra([fit_dialog.fitted_spectrum, fit_dialog.residual_spectrum])
            self.state_changed.emit()

        fit_dialog = FitWidget(self.main_widget.var_widget, accepted, selected_node, parent=self)

    def set_style(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        selected_node = self.myModel.node_from_index(self.selectedIndexes()[0])
        if isinstance(selected_node, SpectrumItemGroup):
            selected_node = selected_node[0]  # select spectrum

        def accepted():
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

            self.redraw_spectra.emit()
            self.state_changed.emit()

        style_widget = StyleWidget(self.main_widget.var_widget, accepted, selected_node, parent=self)

    def rename(self):

        items_count = len(self.selectedIndexes()) / 2

        if items_count == 0:
            return

        if RenameDialog.is_opened:
            RenameDialog.get_instance().activateWindow()
            RenameDialog.get_instance().setFocus()
            return

        expression, offset = Settings.last_rename_expression, 0

        rename_dialog = RenameDialog(expression, offset,
                                     last_rename_take_name_from_list=Settings.last_rename_take_name_from_list)
        if not rename_dialog.accepted:
            return

        if rename_dialog.is_renaming_by_expression:
            expression, offset = rename_dialog.result
        else:
            import csv
            splitted_list = csv.reader([rename_dialog.list], doublequote=True, skipinitialspace=True,
                                       delimiter=',').__next__()

            if len(splitted_list) == 0:
                return

        try:
            i = 0
            for item in self.myModel.iterate_selected_items(skip_groups=True,
                                                            skip_childs_in_selected_groups=False):
                name = item.name
                if rename_dialog.is_renaming_by_expression:
                    name = smart_rename(expression, name, offset)
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

            Settings.last_rename_expression = expression
            Settings.last_rename_take_name_from_list = not rename_dialog.is_renaming_by_expression
            Settings.save()
        except Exception as ex:
            Logger.message(ex.__str__())
            QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)

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
                add_to_group = menu.addAction("Add Items to Group")
                add_to_group.triggered.connect(self.add_selected_items_to_group)

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

    def import_spectra(self, spectra):
        if spectra is None:
            return

        if not isinstance(spectra, list):
            raise ValueError("Argument spectra must be type of list.")

        if len(spectra) == 0:
            return

        add_rows_count = 0

        for node in spectra:
            if isinstance(node, Spectrum):
                # child = SpectrumItem(node, node.name, '', parent=self.myModel.root)
                child = SpectrumItem.init(node, self.myModel.root)

                add_rows_count += 1
            if isinstance(node, list):  # list of spectra
                group_item = SpectrumItemGroup(node[0].group_name, '', parent=self.myModel.root)
                add_rows_count += 1
                for sp in node:
                    # child = SpectrumItem(sp, sp.name, '', parent=group_item)
                    child = SpectrumItem.init(sp, group_item)

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

        spectra = parse_files(filepaths)
        self.import_spectra(spectra)
