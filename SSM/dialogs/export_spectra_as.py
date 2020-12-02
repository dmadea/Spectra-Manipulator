from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from PyQt5.QtCore import Qt
from .gui_export_spectra_as import Ui_Dialog

from pathlib import Path

import os

from ..settings import Settings


class ExportSpectraAsDialog(QtWidgets.QDialog, Ui_Dialog):
    # static variables
    is_opened = False
    _instance = None

    def __init__(self, parent=None):
        super(ExportSpectraAsDialog, self).__init__(parent)
        self.setupUi(self)

        # disable resizing of the window,
        # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        self.setWindowTitle("Export Selected Spectra As")

        self.leDir.setText(Settings.export_spectra_as_dialog_path)
        self.leFileExt.setText(Settings.export_spectra_as_dialog_ext)
        self.leDelimiter.setText(self.textualize_special_chars(Settings.export_spectra_as_dialog_delimiter))
        self.leDecimalSeparator.setText(Settings.export_spectra_as_dialog_decimal_sep)

        self.btnDir.clicked.connect(self.btnDir_clicked)

        self.accepted = False
        self.result = None

        ExportSpectraAsDialog.is_opened = True
        ExportSpectraAsDialog._instance = self

        self.show()
        self.exec()

    def btnDir_clicked(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.leDir.text())
        if dir != '':
            self.leDir.setText(dir)

    @staticmethod
    def get_instance():
        return ExportSpectraAsDialog._instance

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

        Settings.export_spectra_as_dialog_path = self.leDir.text()
        Settings.export_spectra_as_dialog_ext = ext
        Settings.export_spectra_as_dialog_delimiter = delimiter
        Settings.export_spectra_as_dialog_decimal_sep = self.leDecimalSeparator.text()

        Settings.save()

        self.accepted = True
        ExportSpectraAsDialog.is_opened = False
        ExportSpectraAsDialog._instance = None
        super(ExportSpectraAsDialog, self).accept()

    def reject(self):
        ExportSpectraAsDialog.is_opened = False
        ExportSpectraAsDialog._instance = None
        super(ExportSpectraAsDialog, self).reject()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = ExportSpectraAsDialog()
    # Dialog.show()
    sys.exit(app.exec_())
