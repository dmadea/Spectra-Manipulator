from PyQt5.QtWidgets import QMenuBar, QAction, QMenu
from settings import Settings
from dialogs.fitwidget import FitWidget


class MenuBar(QMenuBar):
    MAX_RECENT_FILES = 20

    def __init__(self, parent=None):
        super(MenuBar, self).__init__(parent=parent)

        # ---File Menu---

        self.file_menu = self.addMenu("&File")

        self.open_project_act = QAction("&Open Project", self)
        self.open_project_act.setShortcut("Ctrl+O")
        self.open_project_act.triggered.connect(self.parent().open_project)
        self.file_menu.addAction(self.open_project_act)

        self.save_project_act = QAction("&Save Project", self)
        self.save_project_act.setShortcut("Ctrl+S")
        self.save_project_act.triggered.connect(self.parent().save_project)
        self.file_menu.addAction(self.save_project_act)

        self.save_project_as_act = QAction("Save Project &As", self)
        self.save_project_as_act.setShortcut("Ctrl+Shift+S")
        self.save_project_as_act.triggered.connect(self.parent().save_project_as)
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
        self.import_files_act.triggered.connect(self.parent().file_menu_import_files)
        # self.import_files.setShortcut("Ctrl+I")
        self.file_menu.addAction(self.import_files_act)

        self.export_selected_spectra_as_act = QAction("&Export Selected Spectra As", self)
        # self.export_selected_spectra_as.setShortcut("Ctrl+E")
        self.export_selected_spectra_as_act.triggered.connect(self.parent().export_selected_spectra_as)
        self.file_menu.addAction(self.export_selected_spectra_as_act)

        self.file_menu.addSeparator()

        self.settings_act = QAction("Se&ttings", self)
        self.settings_act.triggered.connect(self.parent().open_settings)
        self.file_menu.addAction(self.settings_act)

        self.file_menu.addSeparator()

        self.exit_act = QAction("E&xit", self)
        self.exit_act.triggered.connect(self.parent().close)
        self.file_menu.addAction(self.exit_act)


        # ---Utilities---

        self.utilities_plot_menu = self.addMenu("&Utilities")

        self.function_plotter_act = QAction("&Function plotter", self)
        self.function_plotter_act.triggered.connect(self.open_function_plotter)
        self.utilities_plot_menu.addAction(self.function_plotter_act)



        # ---Export Plot----

        self.export_plot_menu = self.addMenu("&Export Plot")

        self.copy_as_image_act = QAction("&Copy As Image", self)
        self.copy_as_image_act.triggered.connect(self.parent().grpView.save_plot_to_clipboard_as_png)
        self.export_plot_menu.addAction(self.copy_as_image_act)

        self.copy_as_svg = QAction("&Copy As SVG", self)
        self.copy_as_svg.triggered.connect(self.parent().grpView.save_plot_to_clipboard_as_svg)
        self.export_plot_menu.addAction(self.copy_as_svg)

        # ---About Menu---

        self.about_menu = self.addMenu("&About")

        self.about_act = QAction("&About", self)
        self.about_act.triggered.connect(self.show_about_window)
        self.about_menu.addAction(self.about_act)

    def open_function_plotter(self):
        def accepted():
            self.parent().tree_widget.import_spectra(fit_widget.plotted_function_spectra)
            self.parent().tree_widget.state_changed.emit()

        fit_widget = FitWidget(self.parent().var_widget, accepted, None, parent=self)

    def open_recent_file(self):
        if self.sender():
            self.parent().open_project(filepath=self.sender().data(), open_dialog=False)

    def show_about_window(self):
        # self.parent().console.setVisible(True)
        if not self.parent().console_button.isChecked():
            self.parent().console_button.toggle()

        about_message = f"""
             
<h3>Simple Spectra Manipulator</h3>
<p>Version:&nbsp;<span style="color: #ff00ff;"><strong>{Settings.__version__}</strong></span></p><p>Last release: <strong><span style="color: #ff0000;">{Settings.__last_release__}</span></strong></p>
<p><span style="color: #000000;">This software is distributed under the <span style="color: #339966;"><strong>MIT open-source licence&nbsp;</strong></span></span></p>
<p>Copyright (c) 2019 Dominik Madea</p>
<p>Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:</p>
<p>The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.</p>
<p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>"""

        self.parent().console.print_html(about_message)
