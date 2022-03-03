import logging
import sys
import os
from PyQt5 import QtCore
from spectramanipulator import windows
import logging

# from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QFileDialog, QWidget, QPushButton, QStatusBar, QLabel, \
    QDockWidget, QApplication

from PyQt5.QtWidgets import QMenuBar, QAction, QMenu


from PyQt5.QtGui import QColor, QFont
from PyQt5 import QtWidgets
import argparse
from spectramanipulator import __version__
from .update import TaskUpdate

import pyqtgraph as pg

from .treewidget import TreeWidget, get_hierarchic_list
from .treeview.model import ItemIterator
from .project import Project
from .settings.settings import Settings
from .logger import Logger, Transcript
from .plotwidget import PlotWidget

from .user_namespace import UserNamespace
from .menubar import MenuBar
from .console import Console
from .misc import get_cmap_color, int_default_color_scheme, int_line_style

from .dialogs.settingsdialog import SettingsDialog
from .treeview.item import SpectrumItemGroup
# from .dataloader import parse_files_specific
# from .spectrum import SpectrumList
from .dialogs.fitwidget import FitWidget
from .dialogs.load_kinetics_dialog import LoadKineticsDialog

from .configtree.colors import CmLibSingleton

# import re
# import numpy as np
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

debug = False

MAX_RECENT_FILES = 20

import cProfile
import pstats


class Main(QMainWindow):
    current_file = None

    def __init__(self, app, filepath=None, parent=None):
        super(Main, self).__init__(parent)

        self.app = app

        self.setWindowTitle("Untitled - Simple Spectra Manipulator")

        w, h = 1000, 600

        # setup window size based on current resolution
        if sys.platform == 'win32':
            from win32api import GetSystemMetrics
            w, h = int(GetSystemMetrics(0) * 0.45), int(GetSystemMetrics(1) * 0.45)

        # print(w, h)
        self.resize(w, h)

        self.console = Console(self.app, self)
        sys.stdout = Transcript()

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
        # self.var_widget.titleBarWidget()

        self.tree_widget = TreeWidget(self.dockTreeWidget)
        self.dockTreeWidget.setWidget(self.tree_widget)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockTreeWidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.var_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)

        # fixing the resize bug https://stackoverflow.com/questions/48119969/qdockwidget-splitter-jumps-when-qmainwindow-resized
        self.resizeDocks([self.dockTreeWidget], [3 * w // 7], Qt.Horizontal)
        # self.resizeDocks([self.var_widget], [250], Qt.Vertical)
        self.resizeDocks([self.console], [h // 3], Qt.Vertical)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)

        self.sett = Settings()
        self.sett.load()

        self.grpView = PlotWidget(self)
        self.setCentralWidget(self.grpView)

        self.createStatusBar()
        self.logger = Logger(self.console.showMessage, self.statusBar().showMessage)

        self.tree_widget.redraw_spectra.connect(self.redraw_all_spectra)
        self.tree_widget.state_changed.connect(self.add_star)

        self.user_namespace = UserNamespace(self)

        self.setMenuBar(MenuBar(self))

        self.update_recent_files()

        # open project or file if opened with an argument
        if filepath is not None:
            self.open_project(filepath=filepath, open_dialog=False)

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

        statusBar.addPermanentWidget(self.console_button)
        self.console_button.isChecked()

    def update_current_file(self, filepath):
        self.current_file = filepath
        head, tail = os.path.split(filepath)
        self.setWindowTitle(os.path.splitext(tail)[0] + ' - Spectra Manipulator')

    def update_recent_files(self):
        # num = min(len(Settings.recent_project_filepaths), len(self.menuBar().recent_file_actions))
        recent_project_filepaths = self.sett['/Private settings/Recent project filepaths']
        num = len(recent_project_filepaths)

        # update all of them
        for i in range(num):
            filepath = recent_project_filepaths[i]
            head, tail = os.path.split(filepath)
            text = os.path.split(head)[1] + '/' + os.path.splitext(tail)[0]
            self.menuBar().recent_file_actions[i].setText(text)
            self.menuBar().recent_file_actions[i].setData(filepath)  # save filepath as data to QAction
            self.menuBar().recent_file_actions[i].setVisible(True)
            self.menuBar().recent_file_actions[i].setStatusTip(filepath)  # show filepath in statusbar when hovered

        for i in range(num, self.menuBar().MAX_RECENT_FILES):  # set invisible the rest
            self.menuBar().recent_file_actions[i].setVisible(False)

    def add_recent_file(self, filepath):
        filepath = filepath.replace('\\', '/')  # replace \ for /

        recent_project_filepaths = self.sett['/Private settings/Recent project filepaths']

        # if there is the same filepath in the list, remove this entry
        if filepath in recent_project_filepaths:
            recent_project_filepaths.remove(filepath)

        recent_project_filepaths.insert(0, filepath)
        self.sett['/Private settings/Recent project filepaths'] = recent_project_filepaths[:self.menuBar().MAX_RECENT_FILES]

        self.sett.save()

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
                self.sett.save()
            if reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()

    def open_settings(self):

        # TODO -> change as bottom

        dialog = SettingsDialog(self)

        def accepted_applied(save=False):
            # print(f'accepted_applied {save=}')
            # TODO -->  fix - is run multiple times
            if save:
                self.sett.save()

            self.grpView.update_settings()
            self.redraw_all_spectra()

        dialog.accepted.connect(lambda: accepted_applied(True))
        dialog.applied.connect(lambda: accepted_applied(False))
        dialog.rejected.connect(lambda: accepted_applied(False))

        dialog.show()

    def batch_load_kinetics(self):
        """Opens a Batch Load Kinetics dialog and then call the function from treewidget"""

        def accepted():
            paths = load_kin_dialog.lwFolders.paths
            if len(paths) == 0:
                return

            try:
                spectra_dir_name = load_kin_dialog.leSpectra.text()
                times_name = load_kin_dialog.leTimes.text()
                blank_name = load_kin_dialog.leBlank.text()
                dt = float(
                    load_kin_dialog.leTimeUnit.text()) if load_kin_dialog.cbKineticsMeasuredByEach.isChecked() else None
                bcorr_range = (float(load_kin_dialog.leBCorr0.text()), float(load_kin_dialog.leBCorr1.text())) \
                    if load_kin_dialog.cbBCorr.isChecked() else None

                cut_range = (float(load_kin_dialog.leCut0.text()), float(load_kin_dialog.leCut1.text())) \
                    if load_kin_dialog.cbCut.isChecked() else None

                for path in paths:
                    self.tree_widget.load_kinetic(path, spectra_dir_name, times_name, blank_name,
                                                  dt=dt, b_corr=bcorr_range, cut=cut_range)
            except Exception as ex:
                Logger.message(f"Unable to load kinetics: {ex.__str__()}")

        load_kin_dialog = LoadKineticsDialog(accepted)
        load_kin_dialog.show()

    @staticmethod
    def _open_file_dialog(caption='Open ...', initial_dir='...', _filter='All Files (*.*)',
                          initial_filter='All Files (*.*)', choose_multiple=False):

        f = QFileDialog.getOpenFileNames if choose_multiple else QFileDialog.getOpenFileName

        filepaths = f(caption=caption,
                      directory=initial_dir,
                      filter=_filter,
                      initialFilter=initial_filter)

        if not choose_multiple and filepaths[0] == '':
            return None

        if choose_multiple and len(filepaths[0]) < 1:
            return None

        return filepaths[0]

    def check_for_updates(self):
        ta = TaskUpdate(self)
        if ta.can_update():
            ta.start()

    def open_project(self, filepath=None, open_dialog=True):

        if open_dialog:
            filepath = self._open_file_dialog("Open project", self.sett['/Private settings/Open project dialog path'],
                                              _filter=f"Project files (*{Settings.PROJECT_EXTENSION});;All Files (*.*)",
                                              initial_filter=f"Project files (*{Settings.PROJECT_EXTENSION})",
                                              choose_multiple=False)

            if filepath is None:
                return

            self.sett['/Private settings/Open project dialog path'] = os.path.dirname(filepath)
            self.sett.save()

        if not os.path.exists(filepath):
            filepath = filepath.replace('\\', '/')
            Logger.message(f"File {filepath} does not exist.")
            if filepath in self.sett['/Private settings/Recent project filepaths']:
                self.sett['/Private settings/Recent project filepaths'].remove(filepath)
                self.sett.save()

            self.update_recent_files()
            return

        project = Project.deserialize(filepath)
        # try:
        # except Exception as err:
        #     Logger.message("Unable to load {}.\n{}".format(filepath, err.__str__()))
        #     return

        if self.tree_widget.top_level_items_count() != 0:
            reply = QMessageBox.question(self, 'Open project', "Do you want to merge the project with current project? "
                                                               "By clicking No, current project will be deleted and replaced.",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                pass
            elif reply == QMessageBox.No:
                # delete all spectra and import new
                self.tree_widget.clear()
                # project.settings.set_settings()  # TODO
            else:
                return

        if not hasattr(project, 'spectra_list') or project.spectra_list is None:  # for backward compatibility
            self.tree_widget.import_project(project.generic_item)
        else:  # for backward compatibility, for old project structure
            self.tree_widget.import_spectra(project.spectra_list)
        self.add_recent_file(filepath)

        self.update_current_file(filepath)

    def save_project_as(self):

        if self.tree_widget.top_level_items_count() == 0:
            return

        _filter = f"Project files (*{Settings.PROJECT_EXTENSION})"

        filepath = QFileDialog.getSaveFileName(caption="Save project",
                                               directory=self.sett['/Private settings/Save project dialog path'] if self.current_file is None else self.current_file,
                                               filter=_filter, initialFilter=_filter)

        if filepath[0] == '':
            return

        Logger.message(f"Saving project to {filepath[0]}")

        self.sett['/Private settings/Save project dialog path'] = os.path.dirname(filepath[0])
        self.sett.save()

        project = Project(generic_item=self.tree_widget.myModel.root)
        project.serialize(filepath[0])

        self.update_current_file(filepath[0])
        self.add_recent_file(filepath[0])
        Logger.message("Done")

    def save_project(self):

        if self.tree_widget.top_level_items_count() == 0:
            return

        if self.current_file is not None and os.path.exists(self.current_file):
            project = Project(generic_item=self.tree_widget.myModel.root)
            project.serialize(self.current_file)

            Logger.status_message(f"Project was saved to {self.current_file}.")

            self.update_current_file(self.current_file)
        else:
            self.save_project_as()

    def file_menu_import_files(self):
        filepaths = self._open_file_dialog("Import files", self.sett['/Private settings/Import files dialog path'],
                                           _filter="Data Files (*.txt, *.TXT, *.csv, *.CSV, *.dx, *.DX);;All Files (*.*)",
                                           initial_filter=f"All Files (*.*)",
                                           choose_multiple=True)

        if filepaths is None:
            return

        self.sett['/Private settings/Import files dialog path'] = os.path.dirname(filepaths[0])
        self.sett.save()

        self.tree_widget.import_files(filepaths)

    def redraw_all_spectra(self):
        self.grpView.clear_plots()
        self.redraw_items()

    def update_items_data(self, items):
        self.grpView.update_items_data(items)

    def redraw_items(self, items=None, remove=None):

        if remove is True:
            for item in items:
                self.grpView.remove(item)

        # gradient_mat = None
        # if Settings.color_scheme == 2:  # user defined
        #     gradient_mat = self.get_user_gradient()
        #     if gradient_mat is None:
        #         Console.showMessage("Cannot plot the spectra, user defined gradient matrix is not correct.")
        #         return

        # item_counter = 0
        # iterate over all checked spectra items and draw them
        # it = QTreeWidgetItemIterator(self.treeWidget, QTreeWidgetItemIterator.Checked)

        # for item in self.tree_widget.myModel.iterate_items(ItemIterator.Checked):
        # iterator = items if items is not None else self.tree_widget.myModel.iterate_items(ItemIterator.Checked)

        group_counter = -1
        last_group = None

        cm = CmLibSingleton()

        cmap = cm.getColorMapByKey(
            self.sett['/Public settings/Plotting/Color and line style/Use gradient colormap/Colormap'])

        start_range = self.sett['/Public settings/Plotting/Color and line style/Use gradient colormap/Colormap/Start range']
        end_range = self.sett['/Public settings/Plotting/Color and line style/Use gradient colormap/Colormap/End range']
        reversed = self.sett['/Public settings/Plotting/Color and line style/Use gradient colormap/Colormap/Reversed']
        lw = self.sett['/Public settings/Plotting/Color and line style/Line width']
        use_grad_cmap = self.sett['/Public settings/Plotting/Color and line style/Use gradient colormap']
        reversed_Z_order = self.sett['/Public settings/Plotting/Color and line style/Reversed Z order']
        same_color_in_groups = self.sett['/Public settings/Plotting/Color and line style/Plot spectra with same color in groups']
        same_ls = self.sett['/Public settings/Plotting/Color and line style/Plot spectra with different line style among groups']

        # logging.warning(f"Use gradient colormap {use_grad_cmap}.")

        checked_items = list(self.tree_widget.myModel.iterate_items(ItemIterator.Checked))

        automatically_spread_cmap = self.sett['/Public settings/Plotting/Color and line style/Use gradient colormap/Colormap/Automatically spread']
        n = self.sett['/Public settings/Plotting/Color and line style/Use gradient colormap/Colormap/Automatically spread/Number of spectra']
        n = len(checked_items) if automatically_spread_cmap else n

        # iterate over all checked items
        # for item_counter, item in enumerate(self.tree_widget.myModel.iterate_items(ItemIterator.Checked)):
        for item_counter, item in enumerate(checked_items):
            if isinstance(item, SpectrumItemGroup):
                # last_group = item
                continue

            # check if we are plotting a new group, if so, increment counter
            if item.is_in_group():
                if last_group != item.parent:
                    last_group = item.parent
                    group_counter += 1

            # if item in self.grpView.plotted_items:  # look for keys in dictionary
            #     continue

            # if the spectra item is part of a group, write the group name in square brackets before the spectra name in legend
            spectrum_name = '<strong>{}</strong>: {}'.format(item.parent.name, item.name) if item.is_in_group() \
                else item.name

            style = int_line_style(group_counter) if same_ls and item.is_in_group() else Qt.SolidLine

            counter = group_counter if same_color_in_groups and item.is_in_group() else item_counter

            if use_grad_cmap:
                color = get_cmap_color(counter, cmap, n, start_range, end_range, reversed)
            else:
                color = int_default_color_scheme(counter)
            try:
                line_alpha = item.line_alpha if hasattr(item, 'line_alpha') else 255
                line_color = color if item.color is None else QColor(*item.color) if isinstance(item.color, (
                    tuple, list)) else QColor(item.color)  # if string - html format or name of color
                line_color.setAlpha(line_alpha)
                pen = pg.mkPen(color=line_color,
                               width=lw if item.line_width is None else item.line_width,
                               style=style if item.line_type is None else item.line_type)
            except AttributeError:
                pen = pg.mkPen(color=color, width=lw, style=style)

            try:
                symbol = item.symbol

                _brush = item.symbol_brush if item.symbol_brush is not None else pen.color().name()
                symbolPen = QColor(*_brush) if isinstance(_brush, (tuple, list)) else QColor(_brush)
                symbolPen.setAlpha(item.sym_brush_alpha)
                _fill = item.symbol_fill if item.symbol_fill is not None else pen.color().name()
                symbolBrush = QColor(*_fill) if isinstance(_fill, (tuple, list)) else QColor(_fill)
                symbolBrush.setAlpha(item.sym_fill_alpha)
                symbol_size = item.symbol_size
            except AttributeError:
                symbol = None
                symbolBrush = None
                symbolPen = None
                symbol_size = None

            self.grpView.plot(item,
                              pen=pen,
                              name=spectrum_name,
                              plot_legend=item.plot_legend if hasattr(item, 'plot_legend') else True,
                              symbolBrush=symbolBrush,
                              symbolPen=symbolPen,
                              symbol=symbol,
                              symbolSize=symbol_size,
                              zValue=item_counter if reversed_Z_order else -item_counter)

        FitWidget.replot()  # replot all fits if FitWidget is active


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


def main():
    global debug
    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    parser = argparse.ArgumentParser('spectra-manipulator')
    parser.add_argument("--debug", action="store_true",
                        help="runs in debug mode (keeps the console visible)")
    parser.add_argument("--version", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("--associate_files", action="store_true",
                        help="associates project files for windows and exits")
    parser.add_argument("--remove_association", action="store_true",
                        help="removes project files association from windows registry and exits")
    parser.add_argument("--create_desktop_shortcut", action="store_true",
                        help="creates a shortcut for a program on users desktop")

    args, unparsed_args = parser.parse_known_args()
    debug = args.debug

    if args.associate_files:
        windows.associate_project_file()
        return

    if args.remove_association:
        windows.remove_project_file_association()
        return

    if args.create_desktop_shortcut:
        windows.create_desktop_shortcut()
        return

    try:
        if sys.platform == 'win32' and not debug:
            windows.set_attached_console_visible(False)

        app = QApplication(sys.argv)

        QCoreApplication.setOrganizationName("Spectra Manipulator")
        QCoreApplication.setOrganizationDomain("FreeTimeCoding")
        QCoreApplication.setApplicationName("Spectra Manipulator")

        # app.setStyle('Windows')
        # pass the filename if provided
        form = Main(app, filepath=unparsed_args[0] if len(unparsed_args) > 0 else None)
        form.show()

        # form.interact()
        # app.lastWindowClosed.connect(app.quit)
        sys.exit(app.exec_())
    finally:
        if sys.platform == 'win32' and not debug:
            windows.set_attached_console_visible(True)

    ### cProfile.run('app.exec_()', 'profdata')
    # cProfile.runctx('app.exec_()', None, locals(), filename='profdata')
    # p = pstats.Stats('profdata')
    # p.sort_stats('cumtime').print_stats(100)


if __name__ == "__main__":
    main()

# print(QtWidgets.QStyleFactory.keys())
