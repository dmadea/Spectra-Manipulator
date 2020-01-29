
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtCore import Qt
from dialogs.gui_interpolate_dialog import Ui_Dialog

from webbrowser import open_new_tab


class InterpolateDialog(QtWidgets.QDialog, Ui_Dialog):

    # static variables
    is_opened = False
    _instance = None

    def __init__(self, parent=None):
        super(InterpolateDialog, self).__init__(parent)
        self.setupUi(self)

        #disable resizing of the window,
        # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        self.spacing = 1
        self.selected_kind = 'linear'

        self.setWindowTitle("Interpolate")
        self.leSpacing.setText(str(self.spacing))

        self.kinds = [
            {'kind': 'linear', 'name': 'Linear'},
            {'kind': 'nearest', 'name': 'Nearest'},
            {'kind': 'previous', 'name': 'Previous'},
            {'kind': 'next', 'name': 'Next'},
            {'kind': 'zero', 'name': 'Spline (zero order)'},
            {'kind': 'slinear', 'name': 'Spline (1st order)'},
            {'kind': 'quadratic', 'name': 'Spline (2nd order)'},
            {'kind': 'cubic', 'name': 'Spline (3rd order)'},
        ]

        self.cbIntepolation.addItems(map(lambda d: d['name'], self.kinds))

        self.label.linkActivated.connect(self.open_url)

        self.accepted = False

        InterpolateDialog.is_opened = True
        InterpolateDialog._instance = self

        self.show()
        self.exec()

    def open_url(self):
        open_new_tab('https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d')

    @staticmethod
    def get_instance():
        return InterpolateDialog._instance

    def accept(self):
        try:
            self.spacing = float(self.leSpacing.text().replace(',', '.').strip())
        except ValueError:
            QMessageBox.warning(self, 'Error', "Cannot parse spacing value into float, please correct it.", QMessageBox.Ok)
            return

        self.selected_kind = self.kinds[self.cbIntepolation.currentIndex()]['kind']
        self.accepted = True
        InterpolateDialog.is_opened = False
        InterpolateDialog._instance = None
        super(InterpolateDialog, self).accept()

    def reject(self):
        InterpolateDialog.is_opened = False
        InterpolateDialog._instance = None
        super(InterpolateDialog, self).reject()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = InterpolateDialog()
    # Dialog.show()
    sys.exit(app.exec_())