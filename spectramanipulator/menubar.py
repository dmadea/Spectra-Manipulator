from PyQt5.QtWidgets import QMenuBar, QAction, QMenu
# from spectramanipulator.settings.settings import Settings
# from spectramanipulator.dialogs.fitwidget import FitWidget
from spectramanipulator import __version__
from functools import partial
from .special_importer import import_DX_HPLC_files, import_UV_HPLC_files, import_LPF_kinetics, import_EEM_Duetta, import_kinetics_Duetta, batch_load_kinetics


class MenuBar(QMenuBar):
    MAX_RECENT_FILES = 20

    def __init__(self, parent=None):
        super(MenuBar, self).__init__(parent=parent)
        
        self.main_window = parent

        # ---File Menu---

        self.file_menu = self.addMenu("&File")
        self.setNativeMenuBar(False)

        self.open_project_act = QAction("&Open Project", self)
        self.open_project_act.setShortcut("Ctrl+O")
        self.open_project_act.triggered.connect(self.main_window.open_project)
        self.file_menu.addAction(self.open_project_act)

        self.save_project_act = QAction("&Save Project", self)
        self.save_project_act.setShortcut("Ctrl+S")
        self.save_project_act.triggered.connect(self.main_window.save_project)
        self.file_menu.addAction(self.save_project_act)

        self.save_project_as_act = QAction("Save Project &As", self)
        self.save_project_as_act.setShortcut("Ctrl+Shift+S")
        self.save_project_as_act.triggered.connect(self.main_window.save_project_as)
        self.file_menu.addAction(self.save_project_as_act)

        self.open_recent_menu = QMenu("Open Recent", self.file_menu)
        self.file_menu.addAction(self.open_recent_menu.menuAction())

        # add blank actions
        self.recent_file_actions = []
        for i in range(self.MAX_RECENT_FILES):
            act = self.open_recent_menu.addAction('')
            act.setVisible(False)
            self.recent_file_actions.append(act)
            act.triggered.connect(self.open_recent_file)

        self.file_menu.addSeparator()

        self.import_files_act = QAction("&Import Files", self)
        self.import_files_act.triggered.connect(self.main_window.file_menu_import_files)
        # self.import_files.setShortcut("Ctrl+I")
        self.file_menu.addAction(self.import_files_act)

        self.import_special_menu = QMenu("Import Special", self.file_menu)
        self.file_menu.addAction(self.import_special_menu.menuAction())

        self.nano_kinetics = QAction("Kinetics from LFP (with T\u2192A conversion)", self)
        self.nano_kinetics.triggered.connect(partial(self.import_spectral_data, import_LPF_kinetics))
        self.import_special_menu.addAction(self.nano_kinetics)

        self.EEM_duetta = QAction('Excitation-Emission Map from Duetta Fluorimeter', self)
        self.EEM_duetta.triggered.connect(partial(self.import_spectral_data, import_EEM_Duetta))
        self.import_special_menu.addAction(self.EEM_duetta)

        self.kinetics_duetta = QAction('Kinetics from Duetta Fluorimeter', self)
        self.kinetics_duetta.triggered.connect(partial(self.import_spectral_data, import_kinetics_Duetta))
        self.import_special_menu.addAction(self.kinetics_duetta)

        self.batch_load_kin = QAction('Batch Load UV-VIS kinetics', self)
        self.batch_load_kin.triggered.connect(lambda: batch_load_kinetics(self.main_window.tree_widget.load_kinetic))

        self.import_special_menu.addAction(self.batch_load_kin)

        self.old_HPLC_chrom = QAction('Old Agilent HPLC chromatogram (*.UV)', self)
        self.old_HPLC_chrom.triggered.connect(partial(self.import_spectral_data, import_UV_HPLC_files))
        self.import_special_menu.addAction(self.old_HPLC_chrom)

        self.new_HPLC_chrom = QAction('New Agilent HPLC chromatogram (*.DX)', self)
        self.new_HPLC_chrom.triggered.connect(partial(self.import_spectral_data, import_DX_HPLC_files))
        self.import_special_menu.addAction(self.new_HPLC_chrom)

        self.export_selected_spectra_as_act = QAction("&Export Selected Items As", self)
        # self.export_selected_spectra_as.setShortcut("Ctrl+E")
        self.export_selected_spectra_as_act.triggered.connect(self.main_window.tree_widget.export_selected_items_as)
        self.file_menu.addAction(self.export_selected_spectra_as_act)

        self.file_menu.addSeparator()

        # On mac, Settings on native menu did not show because the name is reserved... WTF
        # https://stackoverflow.com/questions/28559730/menu-entry-not-shown-on-mac-os
        self.settings_act = QAction("Settings", self)
        self.settings_act.triggered.connect(self.main_window.open_settings)
        self.file_menu.addAction(self.settings_act)

        self.file_menu.addSeparator()

        self.exit_act = QAction("E&xit", self)
        self.exit_act.triggered.connect(self.main_window.close)
        self.file_menu.addAction(self.exit_act)

        # ---Utilities---

        self.utilities_plot_menu = self.addMenu("&Utilities")

        self.function_plotter_act = QAction("&Function plotter", self)
        self.function_plotter_act.triggered.connect(self.open_function_plotter)
        self.utilities_plot_menu.addAction(self.function_plotter_act)

        # ---Export Plot----

        # self.export_plot_menu = self.addMenu("&Export Plot")
        #
        # self.copy_as_image_act = QAction("&Copy As Image", self)
        # self.copy_as_image_act.triggered.connect(self.main_window.grpView.save_plot_to_clipboard_as_png)
        # self.export_plot_menu.addAction(self.copy_as_image_act)
        #
        # self.copy_as_svg = QAction("&Copy As SVG", self)
        # self.copy_as_svg.triggered.connect(self.main_window.grpView.save_plot_to_clipboard_as_svg)
        # self.export_plot_menu.addAction(self.copy_as_svg)

        # ---About Menu---

        self.about_menu = self.addMenu("&Help")

        self.about_act = QAction("&About", self)
        self.about_act.triggered.connect(self.show_about_window)
        self.about_menu.addAction(self.about_act)

        self.update_act = QAction("&Check for updates...", self)
        self.update_act.triggered.connect(self.main_window.check_for_updates)
        self.about_menu.addAction(self.update_act)

    def import_spectral_data(self, func):
        spectral_data = func()
        if spectral_data is None:
            return
        self.main_window.tree_widget.import_spectra(spectral_data)

    def open_function_plotter(self):
        # TODO
        return
        # def accepted():
        #     self.main_window.tree_widget.import_spectra(fit_widget.plotted_function_spectra)
        #     self.main_window.tree_widget.state_changed.emit()
        #
        # fit_widget = FitWidget(self.main_window.var_widget, accepted, None, parent=self)

    def open_recent_file(self):
        if self.sender():
            self.main_window.open_project(filepath=self.sender().data(), open_dialog=False)

    def show_about_window(self):
        # self.main_window.console.setVisible(True)
        if not self.main_window.console_button.isChecked():
            self.main_window.console_button.toggle()

        about_message = f"""
             
<h3>Simple Spectra Manipulator</h3>
<p>Version:&nbsp;<span style="color: #ff00ff;"><strong>{__version__}</strong></span></p>
<p><span style="color: #000000;">This software is distributed under the <span style="color: #339966;"><strong>MIT open-source licence&nbsp;</strong></span></span></p>
<p>Copyright (c) 2020 Dominik Madea</p>
<p>Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:</p>
<p>The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.</p>
<p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>"""

        self.main_window.console.print_html(about_message)
