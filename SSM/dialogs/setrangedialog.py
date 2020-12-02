
from PyQt5 import QtCore, QtGui, QtWidgets

# from PyQt5 import *
from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import *
from .gui_setrangedialog import Ui_Dialog
import sys


class SetRangeDialog(QtWidgets.QDialog, Ui_Dialog):

    # static variables
    is_opened = False
    _instance = None

    def __init__(self, x0_value=0.0, x1_value=100.0, title='SetRangeDialog', label='Set xrange',  parent=None):
        super(SetRangeDialog, self).__init__(parent)
        self.setupUi(self)

        #disable resizing of the window,
        # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        self.setWindowTitle(title)
        self.label.setText(label)

        # self.dbX0.valueChanged.connect(self.dbX0_value_changed)
        # self.dbX1.valueChanged.connect(self.dbX1_value_changed)

        self.dbX0.setValue(x0_value)
        self.dbX1.setValue(x1_value)

        self.accepted = False

        SetRangeDialog.is_opened = True
        SetRangeDialog._instance = self

        self.dbX0.setFocus()
        self.dbX0.selectAll()

        self.show()
        self.exec()
        # sys.exit(self.exec_())

    @staticmethod
    def get_instance():
        return SetRangeDialog._instance

    def set_result(self):
        self.returned_range = (self.dbX0.value(), self.dbX1.value())

    def accept(self):
        self.set_result()
        self.accepted = True
        SetRangeDialog.is_opened = False
        SetRangeDialog._instance = None
        super(SetRangeDialog, self).accept()

    def reject(self):
        SetRangeDialog.is_opened = False
        SetRangeDialog._instance = None
        super(SetRangeDialog, self).reject()

    # def dbX0_value_changed(self, *args):
    #     self.dbX1.setMinimum(self.dbX0.value())
    #
    # def dbX1_value_changed(self, *args):
    #     self.dbX0.setMaximum(self.dbX1.value())

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = SetRangeDialog()
    Dialog.show()
    sys.exit(app.exec_())