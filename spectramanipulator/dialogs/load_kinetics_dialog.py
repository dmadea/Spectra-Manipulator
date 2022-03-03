
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QListView, QAbstractItemView, QTreeView, QListWidget, QMessageBox, QLabel
from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QHBoxLayout, QGridLayout, QToolButton, QCheckBox, QDialogButtonBox

from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from spectramanipulator.settings.settings import Settings
from spectramanipulator.singleton import PersistentDialog
from typing import Callable
import os


class MyQListWidget(QListWidget):

    def __init__(self, parent=None):
        super(MyQListWidget, self).__init__(parent=parent)
        self.setSelectionMode(QAbstractItemView.MultiSelection)

        self.item_names = []  # shortcut paths
        self.paths = []  # directory paths

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == 16777223:  # del pressed
            for item in self.selectedItems():
                index = self.item_names.index(item.text())
                del self.item_names[index]
                del self.paths[index]
                self.takeItem(self.row(item))

    def addItem(self, aitem, path) -> None:
        super(MyQListWidget, self).addItem(aitem)
        self.item_names.append(aitem)
        self.paths.append(path)


class LoadKineticsDialog(PersistentDialog):

    def __init__(self, accept_func: Callable, parent=None):
        super(LoadKineticsDialog, self).__init__(parent)

        self.setWindowTitle("Batch Load Kinetics")
        self.sett = Settings()

        self.description = QLabel("""
Loads UV-Vis kinetics. The spectra can be in various formats (dx, csv, txt, etc.). If blank spectrum is provided, it will be subtracted from each of the spectra. Time corresponding to each spectrum will be set as the name for that spectrum. Times can be provided by time difference (in case kinetics was measured reguraly) or from filename. Postprocessing (baseline correction and/or cut of spectra) can be performed on loaded dataset.
Required folder structure for one kinetics:
[kinetics folder]
    [spectra folder]
         01.dx
         02.dx
         ...
    times.txt (optional)
    blank.dx (optional)
        
""")
        self.description.setWordWrap(True)
        self.accept_func = accept_func

        self.btnChooseDirs = QToolButton()
        self.btnChooseDirs.setText('...')

        self.layout_button = QHBoxLayout()
        self.layout_button.addWidget(QLabel('Folders to load:'))
        self.layout_button.addWidget(self.btnChooseDirs)

        self.lwFolders = MyQListWidget(self)

        self.grid_layout = QGridLayout()
        self.lbl1 = QLabel('Spectra folder name:')
        self.lbl2 = QLabel('Blank spectrum name (optional):')
        self.cbKineticsMeasuredByEach = QCheckBox('Kinetics measured by each (time unit):')
        self.cbKineticsMeasuredByEach.setChecked(True)
        self.lbl4 = QLabel('Use times from filename (optional):')
        self.cbBCorr = QCheckBox('Apply baseline correction in range:')
        self.cbCut = QCheckBox('Cut spectra to range:')

        self.leSpectra = QLineEdit('spectra')
        self.leBlank = QLineEdit('blank.dx')
        self.leTimeUnit = QLineEdit('1')
        self.leTimes = QLineEdit('times.txt')
        self.leBCorr0 = QLineEdit('700')
        self.leBCorr1 = QLineEdit('800')
        self.leCut0 = QLineEdit('230')
        self.leCut1 = QLineEdit('650')

        self.Hlayout1 = QHBoxLayout()
        self.Hlayout1.addWidget(self.leBCorr0)
        self.Hlayout1.addWidget(QLabel('to'))
        self.Hlayout1.addWidget(self.leBCorr1)

        self.Hlayout2 = QHBoxLayout()
        self.Hlayout2.addWidget(self.leCut0)
        self.Hlayout2.addWidget(QLabel('to'))
        self.Hlayout2.addWidget(self.leCut1)

        self.grid_layout.addWidget(self.lbl1, 0, 0, 1, 1)
        self.grid_layout.addWidget(self.lbl2, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.cbKineticsMeasuredByEach, 2, 0, 1, 1)
        self.grid_layout.addWidget(self.lbl4, 3, 0, 1, 1)
        self.grid_layout.addWidget(self.cbBCorr, 4, 0, 1, 1)
        self.grid_layout.addWidget(self.cbCut, 5, 0, 1, 1)
        self.grid_layout.addWidget(self.leSpectra, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.leBlank, 1, 1, 1, 1)
        self.grid_layout.addWidget(self.leTimeUnit, 2, 1, 1, 1)
        self.grid_layout.addWidget(self.leTimes, 3, 1, 1, 1)
        self.grid_layout.addLayout(self.Hlayout1, 4, 1, 1, 1)
        self.grid_layout.addLayout(self.Hlayout2, 5, 1, 1, 1)

        self.button_box = QDialogButtonBox()
        self.button_box.setOrientation(Qt.Horizontal)
        self.button_box.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)  # OK button
        self.button_box.rejected.connect(self.reject)  # Cancel button

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.description)
        self.main_layout.addLayout(self.layout_button)
        self.main_layout.addWidget(self.lwFolders)
        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addWidget(self.button_box)

        self.setLayout(self.main_layout)

        self.cbKineticsMeasuredByEach.toggled.connect(self.cbKineticsMeasuredByEach_toggled)
        self.cbBCorr.toggled.connect(self.cbBCorr_toggled)
        self.cbCut.toggled.connect(self.cbCut_toggled)
        self.btnChooseDirs.clicked.connect(self.btnChooseDirs_clicked)

        self.cbKineticsMeasuredByEach_toggled()
        self.cbBCorr_toggled()
        self.cbCut_toggled()

    def btnChooseDirs_clicked(self):
        # https://stackoverflow.com/questions/38252419/how-to-get-qfiledialog-to-select-and-return-multiple-folders
        # just copied :)
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_dialog.setDirectory(self.sett['/Private settings/Load kinetics last path'])
        file_view = file_dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            for path in file_dialog.selectedFiles():
                head, tail = os.path.split(path)
                head2, tail2 = os.path.split(head)
                name = os.path.join(tail2, tail)
                self.sett['/Private settings/Load kinetics last path'] = head
                if name not in self.lwFolders.item_names:
                    self.lwFolders.addItem(name, path)

    def cbBCorr_toggled(self):
        checked = self.cbBCorr.isChecked()
        self.leBCorr0.setEnabled(checked)
        self.leBCorr1.setEnabled(checked)

    def cbCut_toggled(self):
        checked = self.cbCut.isChecked()
        self.leCut0.setEnabled(checked)
        self.leCut1.setEnabled(checked)

    def cbKineticsMeasuredByEach_toggled(self):
        checked = self.cbKineticsMeasuredByEach.isChecked()
        self.leTimeUnit.setEnabled(checked)
        self.leTimes.setEnabled(not checked)

    def accept(self):
        try:
            float(self.leTimeUnit.text())
            float(self.leCut0.text())
            float(self.leCut1.text())
            float(self.leBCorr0.text())
            float(self.leBCorr1.text())
        except ValueError:
            QMessageBox.critical(self, 'Error', "Invalid input, please check the fields.")
            return

        self.sett.save()
        self.accept_func()
        super(LoadKineticsDialog, self).accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = LoadKineticsDialog()
    Dialog.show()
    sys.exit(app.exec_())


