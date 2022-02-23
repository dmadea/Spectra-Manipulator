from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QLineEdit, QDialogButtonBox, QToolButton, QVBoxLayout, QGridLayout, QHBoxLayout

from PyQt5.QtCore import Qt
from typing import Callable
# from .gui_export_spectra_as import Ui_Dialog

from pathlib import Path

import os

from spectramanipulator.settings.settings import Settings
from spectramanipulator.singleton import PersistentDialog


class ExportSpectraAsDialog(PersistentDialog):

    def __init__(self, accepted_func: Callable, parent=None):
        super(ExportSpectraAsDialog, self).__init__(parent)

        self.sett = Settings()
        self.accepted_func = accepted_func
        self.setWindowTitle("Export Selected Spectra As")

        self.description_label = QLabel("Each selected top-level item will be saved in separate file with its name as filename. Spectra in groups will be concatenated. Select directory where spectra will be saved and specify the file extension. Non existing directories will be created. Top level items with same name will be overwritten. Delimiter field: use \\t for tabulator.")
        self.description_label.setWordWrap(True)

        self.leDir = QLineEdit()
        self.leFileExt = QLineEdit()
        self.leDelimiter = QLineEdit()
        self.leDecimalSeparator = QLineEdit()

        self.btnDir = QToolButton()
        self.btnDir.setText('...')
        self.btnDir.clicked.connect(self.btnDir_clicked)
        self.label1 = QLabel('File extension:')
        self.label2 = QLabel('Delimiter:')
        self.label3 = QLabel('Decimal separator:')

        self.button_box = QDialogButtonBox(self)
        self.button_box.setOrientation(Qt.Horizontal)
        self.button_box.setStandardButtons(QDialogButtonBox.Save | QDialogButtonBox.Cancel)

        self.button_box.accepted.connect(self.accept)  # OK button
        self.button_box.rejected.connect(self.reject)  # Cancel button

        self.grid_layout = QGridLayout()

        self.grid_layout.addWidget(self.label1, 0, 0, 1, 1)
        self.grid_layout.addWidget(self.label2, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.label3, 2, 0, 1, 1)
        self.grid_layout.addWidget(self.leFileExt, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.leDelimiter, 1, 1, 1, 1)
        self.grid_layout.addWidget(self.leDecimalSeparator, 2, 1, 1, 1)

        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.leDir)
        self.h_layout.addWidget(self.btnDir)

        self.h2_layout = QHBoxLayout()
        self.h2_layout.addStretch()
        self.h2_layout.addWidget(self.button_box)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.description_label)
        self.main_layout.addLayout(self.h_layout)
        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.h2_layout)

        self.setLayout(self.main_layout)

        self.leDir.setText(self.sett['/Private settings/Export spectra as dialog/Path'])
        self.leFileExt.setText(self.sett['/Private settings/Export spectra as dialog/Ext'])
        self.leDelimiter.setText(self.textualize_special_chars(self.sett['/Private settings/Export spectra as dialog/Delimiter']))
        self.leDecimalSeparator.setText(self.sett['/Private settings/Export spectra as dialog/Decimal separator'])
        print('__init__ called')

        self.accepted = False
        self.result = None

    def btnDir_clicked(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.leDir.text())
        if dir != '':
            self.leDir.setText(dir)

    @staticmethod
    def textualize_special_chars(text):
        return text.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

    @staticmethod
    def DEtextualize_special_chars(text):
        return text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

    def accept(self):

        if self.leDir.text() == '':
            QMessageBox.warning(self, 'Information', "Directory cannot be empty", QMessageBox.Ok)
            return

        try:
            path = Path(self.leDir.text())
            path.mkdir(parents=True, exist_ok=True)

            if not os.path.isdir(self.leDir.text()):
                raise Exception
        except Exception as ex:
            QMessageBox.warning(self, 'Information',
                                "Please check the directory path.\n{}".format(ex.__str__()),
                                QMessageBox.Ok)
            return

        if self.leDecimalSeparator.text() == '' or self.leFileExt.text() == '' or self.leDelimiter.text() == '':
            QMessageBox.warning(self, 'Information', "Fields cannot be empty.", QMessageBox.Ok)
            return

        ext = self.leFileExt.text()
        if not ext.startswith('.'):
            ext = '.' + ext

        delimiter = self.DEtextualize_special_chars(self.leDelimiter.text())

        self.result = (
            self.leDir.text(),
            ext.lower(),
            delimiter,
            self.leDecimalSeparator.text()
        )

        self.sett['/Private settings/Export spectra as dialog/Path'] = self.leDir.text()
        self.sett['/Private settings/Export spectra as dialog/Ext'] = ext
        self.sett['/Private settings/Export spectra as dialog/Delimiter'] = delimiter
        self.sett['/Private settings/Export spectra as dialog/Decimal separator'] = self.leDecimalSeparator.text()

        self.sett.save()

        self.accepted = True
        self.accepted_func()
        super(ExportSpectraAsDialog, self).accept()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = ExportSpectraAsDialog()
    Dialog.show()
    sys.exit(app.exec_())
