import sys
import os
from PyQt5 import QtCore
# from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QFileDialog, QWidget, QPushButton, QStatusBar, QLabel, \
    QDockWidget, QApplication
from PyQt5.QtGui import QColor, QFont
from PyQt5 import QtWidgets

import pyqtgraph as pg

from treewidget import SpectrumItemGroup, TreeWidget
from treeview.model import ItemIterator
from project import Project
from settings import Settings
from logger import Logger
from plotwidget import PlotWidget
from spectrum import Spectrum

from user_namespace import UserNamespace
from menubar import MenuBar
from console import Console

from dialogs.settingsdialog import SettingsDialog
from dialogs.export_spectra_as import ExportSpectraAsDialog
import numpy as np

import cProfile
import pstats


class Main(QMainWindow):
    current_file = None

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        self.setWindowTitle("Untitled - Simple Spectra Manipulator")

        self.resize(1000, 600)

        self.console = Console(self)

        self.dockTreeWidget = QDockWidget(self)
        self.dockTreeWidget.setTitleBarWidget(QWidget())
        self.dockTreeWidget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dockTreeWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)

        # create and set variable docking widget, where different function widows will appear
        self.var_widget = QDockWidget(self)
        lbl = QLabel('Some Title Text')
        lbl.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        lbl.setFont(font)
        self.var_widget.setTitleBarWidget(lbl)
        self.var_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.var_widget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        self.var_widget.setVisible(False)
        self.var_widget.titleBarWidget()

        self.tree_widget = TreeWidget(self.dockTreeWidget)
        self.dockTreeWidget.setWidget(self.tree_widget)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockTreeWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.var_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)
        # fixing the resize bug https://stackoverflow.com/questions/48119969/qdockwidget-splitter-jumps-when-qmainwindow-resized
        self.resizeDocks([self.dockTreeWidget], [270], Qt.Horizontal)
        # self.resizeDocks([self.var_widget], [250], Qt.Vertical)
        self.resizeDocks([self.console], [200], Qt.Vertical)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)

        self.coor_label = QLabel()
        self.grpView = PlotWidget(self, coordinates_func=self.coor_label.setText)
        self.setCentralWidget(self.grpView)

        self.createStatusBar()
        self.logger = Logger(self.console.showMessage, self.statusBar().showMessage)

        # self.treeWidget.itemCheckStateChanged.connect(self.item_checked_changed)
        # self.treeWidget.itemEdited.connect(self.item_edited)
        # self.treeWidget.itemsDeleted.connect(self.items_deleted)
        self.tree_widget.redraw_spectra.connect(self.redraw_all_spectra)
        self.tree_widget.state_changed.connect(self.add_star)

        self.user_namespace = UserNamespace(self)

        self.setMenuBar(MenuBar(self))

        Settings.load()

        self.grpView.update_settings()

        self.update_recent_files()

    def add_star(self):
        if not self.windowTitle().startswith('*'):
            self.setWindowTitle('*' + self.windowTitle())

    def createStatusBar(self):
        statusBar = QStatusBar()
        self.setStatusBar(statusBar)
        statusBar.showMessage("Ready", 3000)
        self.console_button = QPushButton("Console")
        self.console_button.setFlat(True)
        self.console_button.setCheckable(True)
        self.console_button.toggled.connect(self.console.setVisible)
        statusBar.addPermanentWidget(self.coor_label)

        statusBar.addPermanentWidget(self.console_button)
        self.console_button.isChecked()

        # self.grpView.coordinates_func = self.lblCoordinates.setText
        #
        # self.lblCoordinates.setText('')

    def update_current_file(self, filepath):
        self.current_file = filepath
        head, tail = os.path.split(filepath)
        self.setWindowTitle(os.path.splitext(tail)[0] + ' - Simple Spectra Manipulator')

    def update_recent_files(self):
        num = min(len(Settings.recent_project_filepaths), len(self.menuBar().recent_file_actions))

        for i in range(num):
            filepath = Settings.recent_project_filepaths[i]
            head, tail = os.path.split(filepath)
            text = os.path.split(head)[1] + '\\' + os.path.splitext(tail)[0]
            self.menuBar().recent_file_actions[i].setText(text)
            self.menuBar().recent_file_actions[i].setData(filepath)
            self.menuBar().recent_file_actions[i].setVisible(True)

    def add_recent_file(self, filepath):
        # if there is the same filepath in the list, remove this entry
        if filepath in Settings.recent_project_filepaths:
            Settings.recent_project_filepaths.remove(filepath)

        Settings.recent_project_filepaths.insert(0, filepath)
        while len(Settings.recent_project_filepaths) > len(self.menuBar().recent_file_actions):
            # remove last one
            Settings.recent_project_filepaths = Settings.recent_project_filepaths[:-1]
        Settings.save()
        self.update_recent_files()

    def actioncopy_to_svg_clicked(self):

        self.grpView.save_plot_to_clipboard_as_svg()

    # on close
    def closeEvent(self, event):
        if self.windowTitle().startswith('*') and self.tree_widget.top_level_items_count() > 0:
            reply = QMessageBox.question(self, 'Message', "Do you want to save the project?", QMessageBox.Yes |
                                         QMessageBox.No | QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                self.save_project()
                Settings.save()
            if reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()

    def export_selected_spectra_as(self):

        if ExportSpectraAsDialog.is_opened:
            ExportSpectraAsDialog.get_instance().activateWindow()
            ExportSpectraAsDialog.get_instance().setFocus()
            return

        if len(self.tree_widget.selectedIndexes()) == 0:
            return

        dialog = ExportSpectraAsDialog()

        if not dialog.accepted:
            return

        path, ext, delimiter, decimal_sep = dialog.result

        sp_list = self.tree_widget.get_hierarchic_list(
            self.tree_widget.myModel.iterate_selected_items(skip_groups=True,
                                                            skip_childs_in_selected_groups=False))

        try:
            Spectrum.list_to_files(sp_list, path, ext, include_group_name=Settings.files_exp_include_group_name,
                                   include_header=Settings.files_exp_include_header,
                                   delimiter=delimiter,
                                   decimal_sep=decimal_sep,
                                   x_data_name=Settings.bottom_axis_label)

        except Exception as ex:
            QMessageBox.warning(self, 'Error', ex.__str__(), QMessageBox.Ok)

        Logger.status_message(f"Data were saved to {path}")

    def open_settings(self):

        if SettingsDialog.is_opened:
            SettingsDialog.get_instance().activateWindow()
            SettingsDialog.get_instance().setFocus()
            return

        sett_dialog = SettingsDialog()

        if not sett_dialog.accepted:
            return

        self.grpView.update_settings()

        self.redraw_all_spectra()

    def open_project(self, filepath=None, open_dialog=True):

        if open_dialog:
            # filter = "Data Files (*.txt, *.csv, *.dx)|*.txt;*.csv;*.dx|All Files (*.*)|*.*"
            filter = "Project files (*.smpj);;All Files (*.*)"
            initial_filter = "Project files (*.smpj)"

            filepaths = QFileDialog.getOpenFileName(caption="Open project",
                                                    directory=Settings.open_project_dialog_path,
                                                    filter=filter,
                                                    initialFilter=initial_filter)
            if filepaths[0] == '':
                return

            Settings.open_project_dialog_path = os.path.split(filepaths[0])[0]
            filepath = filepaths[0]

        try:
            project = Project.deserialize(filepath)
        except:
            return

        if self.tree_widget.top_level_items_count() != 0:
            reply = QMessageBox.question(self, 'Open project', "Do you want to merge the project with current project? "
                                                               "By clicking No, current project will be deleted and replaced.",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                pass
            elif reply == QMessageBox.No:
                # delete all spectra and import new
                self.tree_widget.clear()
                project.settings.set_settings()
            else:
                return

        self.tree_widget.import_spectra(project.spectra_list)
        self.add_recent_file(filepath)

        self.update_current_file(filepath)

    def save_project_as(self):

        if self.tree_widget.top_level_items_count() == 0:
            return

        filter = "Project files (*.smpj)"

        filepath = QFileDialog.getSaveFileName(caption="Save project",
                                               directory=Settings.save_project_dialog_path if self.current_file is None else self.current_file,
                                               filter=filter, initialFilter=filter)

        if filepath[0] == '':
            return

        Logger.message(f"Saving project to {filepath[0]}")

        Settings.save_project_dialog_path = os.path.split(filepath[0])[0]
        Settings.save()

        sp_list = self.tree_widget.get_hierarchic_list(
            self.tree_widget.myModel.iterate_items(ItemIterator.NoChildren))

        project = Project(sp_list)
        project.serialize(filepath[0])

        self.update_current_file(filepath[0])
        self.add_recent_file(filepath[0])
        Logger.message("Done")

    def save_project(self):

        if self.tree_widget.top_level_items_count() == 0:
            return

        if self.current_file is not None:
            sp_list = self.tree_widget.get_hierarchic_list(
                self.tree_widget.myModel.iterate_items(ItemIterator.NoChildren))

            project = Project(sp_list)
            project.serialize(self.current_file)

            Logger.status_message(f"Project was saved to {self.current_file}.")

            self.update_current_file(self.current_file)
        else:
            self.save_project_as()

    def file_menu_import_files(self):

        filter = "Data Files (*.txt, *.TXT, *.csv, *.CSV, *.dx, *.DX);;All Files (*.*)"
        initial_filter = "All Files (*.*)"

        filenames = QFileDialog.getOpenFileNames(None, caption="Import files",
                                                 directory=Settings.import_files_dialog_path,
                                                 filter=filter, initialFilter=initial_filter)

        if len(filenames[0]) == 0:
            return

        Settings.import_files_dialog_path = os.path.split(filenames[0][0])[0]

        self.tree_widget.import_files(filenames[0])

    def intLineStyle(self, counter):
        styles = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine, Qt.DashDotDotLine]
        return styles[counter % len(styles)]

    @staticmethod
    def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255,
                 reversed=False):
        """
        Creates a QColor from a single index. Useful for stepping through a predefined list of colors.

        The argument *index* determines which color from the set will be returned. All other arguments determine what the set of predefined colors will be

        Colors are chosen by cycling across hues while varying the value (brightness).
        By default, this selects from a list of 9 hues."""
        hues = int(hues)
        values = int(values)
        ind = int(index) % (hues * values)
        indh = ind % hues
        indv = ind // hues
        if values > 1:
            v = minValue + indv * ((maxValue - minValue) / (values - 1))
        else:
            v = maxValue

        if reversed:
            h = minHue + ((hues - indh - 1) * (maxHue - minHue)) / hues
        else:
            h = minHue + (indh * (maxHue - minHue)) / hues

        c = QColor()
        c.setHsv(h, sat, v)
        c.setAlpha(alpha)
        return c

    def get_user_gradient(self):
        """Gradient in a format of
        position (0, 1) \t R \t G \t B \t A \n
        etc.

        Entries separated by tabulator \t and lines by new line \n

        """
        try:
            lines = Settings.user_defined_grad.split('\n')
            lines = list(filter(None, lines))  # remove empty entries

            data = np.zeros((len(lines), 5), dtype=np.float32)

            for i, line in enumerate(lines):  # parse the string data into matrix
                entries = line.split('\t')
                data[i] = np.asarray([float(entry) for entry in entries])

            data[:, 1:] *= 255  # multiply the rgba values by 255

            return data
        except Exception :
            pass
            # Console.showMessage("User defined color scheme is not correct.")

    @staticmethod
    def intColorGradient(index, hues, grad_mat, reversed=False):

        ind = int(index) % hues

        pos = ((hues - ind) / hues) if reversed else (ind / hues)

        positions = grad_mat[:, 0].flatten()
        idx_pos = np.searchsorted(positions, pos, side="right")
        idx_pos -= 1 if idx_pos > 0 else 0
        idx_pos -= 1 if idx_pos == len(positions) - 1 else 0

        # position within the interval of colors
        x = (pos - positions[idx_pos]) / (positions[idx_pos + 1] - positions[idx_pos])

        # calculate the resulting color as a linear combination of colors in the interval
        color_vector = (1 - x) * grad_mat[idx_pos, 1:] + x * grad_mat[idx_pos + 1, 1:]  # RGBA

        color = QColor(*color_vector)
        # color.setAlpha(int(color_vector[-1]))
        return color

    @staticmethod
    def int_default_color_scheme(counter):
        colors = [
            (255, 0, 0, 255),  # red
            (0, 255, 0, 255),  # green
            (0, 0, 255, 255),  # blue
            (0, 0, 0, 255),  # black
            (255, 255, 0, 255),  # yellow
            (255, 0, 255, 255),  # magenta
            (0, 255, 255, 255),  # cyan
            (155, 155, 155, 255),  # gray
            (155, 0, 0, 255),  # dark red
            (0, 155, 0, 255),  # dark green
            (0, 0, 155, 255),  # dark blue
            (155, 155, 0, 255),  # dark yellow
            (155, 0, 155, 255),  # dark magenta
            (0, 155, 155, 255)  # dark cyan
        ]

        return QColor(*colors[counter % len(colors)])

    def redraw_all_spectra(self):
        self.grpView.plotItem.clearPlots()

        try:
            # self.grpView.plotItem.legend.scene().removeItem(self.grpView.plotItem.legend)
            self.grpView.legend.scene().removeItem(self.grpView.plotItem.legend)
        except Exception as e:
            print(e)

        gradient_mat = None
        if Settings.color_scheme == 2:  # user defined
            gradient_mat = self.get_user_gradient()
            if gradient_mat is None:
                Console.showMessage("Cannot plot the spectra, user defined gradient matrix is not correct.")
                return

        # self.grpView.plotItem.addLegend(offset=(-30, 30))
        self.grpView.add_legend(spacing=Settings.legend_spacing, offset=(-30, 30))

        item_counter = 0
        group_counter = -1
        # iterate over all checked spectra items and draw them
        # it = QTreeWidgetItemIterator(self.treeWidget, QTreeWidgetItemIterator.Checked)

        last_group = None

        for item in self.tree_widget.myModel.iterate_items(ItemIterator.Checked):

            if isinstance(item, SpectrumItemGroup):
                # last_group = item
                continue

            sp = item
            # if the spectra item is part of a group, write the group name in square brackets before the spectra name in legend
            spectrum_name = '<strong>{}</strong>: {}'.format(item.parent.name, item.name) if item.is_in_group() \
                else item.name

            # check if we are plotting a new group, if so, increment counter
            if item.is_in_group():
                if last_group != item.parent:
                    last_group = item.parent
                    group_counter += 1

            style = self.intLineStyle(
                group_counter) if Settings.different_line_style_among_groups and item.is_in_group() else Qt.SolidLine

            counter = group_counter if Settings.same_color_in_group and item.is_in_group() else item_counter

            if Settings.color_scheme == 0:
                color = self.int_default_color_scheme(counter)
            elif Settings.color_scheme == 1:
                color = self.intColor(counter,
                                      hues=Settings.hues,
                                      values=Settings.values,
                                      maxValue=Settings.maxValue,
                                      minValue=Settings.minValue,
                                      maxHue=Settings.maxHue,
                                      minHue=Settings.minHue,
                                      sat=Settings.sat,
                                      alpha=Settings.alpha,
                                      reversed=Settings.HSV_reversed)
            else:
                color = self.intColorGradient(counter, Settings.hues, gradient_mat, reversed=Settings.HSV_reversed)

            try:
                line_alpha = sp.line_alpha if hasattr(sp, 'line_alpha') else 255
                line_color = color if sp.color is None else QColor(*sp.color) if isinstance(sp.color, (
                    tuple, list)) else QColor(sp.color)  # if string - html format or name of color
                if Settings.color_scheme < 2:  # only for default and HSV
                    line_color.setAlpha(line_alpha)
                pen = pg.mkPen(color=line_color,
                               width=Settings.line_width if sp.line_width is None else sp.line_width,
                               style=style if sp.line_type is None else sp.line_type)
            except AttributeError:
                pen = pg.mkPen(color=color, width=Settings.line_width, style=style)

            try:
                symbol = sp.symbol

                _brush = sp.symbol_brush if sp.symbol_brush is not None else pen.color().name()
                symbolPen = QColor(*_brush) if isinstance(_brush, (tuple, list)) else QColor(_brush)
                symbolPen.setAlpha(sp.sym_brush_alpha)
                _fill = sp.symbol_fill if sp.symbol_fill is not None else pen.color().name()
                symbolBrush = QColor(*_fill) if isinstance(_fill, (tuple, list)) else QColor(_fill)
                symbolBrush.setAlpha(sp.sym_fill_alpha)
                symbol_size = sp.symbol_size
            except AttributeError:
                symbol = None
                symbolBrush = None
                symbolPen = None
                symbol_size = None

            self.grpView.plotItem.plot(sp.data,
                                       pen=pen,
                                       name=spectrum_name,
                                       plot_legend=sp.plot_legend if hasattr(sp, 'plot_legend') else True,
                                       symbolBrush=symbolBrush,
                                       symbolPen=symbolPen,
                                       symbol=symbol,
                                       symbolSize=symbol_size,
                                       zValue=item_counter if Settings.reverse_z_order else (1e5 - item_counter))

            # Console.showMessage(spectrum_name + "  " + str(plot.zValue()))
            item_counter += 1


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


if __name__ == "__main__":
    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    app = QApplication(sys.argv)
    # app.setStyle('Windows')
    form = Main()
    form.show()

    # form.interact()

    sys.exit(app.exec_())

    # cProfile.run('app.exec_()', 'profdata')
    # cProfile.runctx('app.exec_()', None, locals())
    # p = pstats.Stats('profdata')
    # p.sort_stats('time').print_stats()

# print(QtWidgets.QStyleFactory.keys())