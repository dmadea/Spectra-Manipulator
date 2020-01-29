
from PyQt5 import QtCore, QtGui, QtWidgets

# from PyQt5 import *
from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import *
from dialogs.gui_intintinputdialog import Ui_Dialog
import sys


class IntIntInputDialog(QtWidgets.QDialog, Ui_Dialog):

    # static variables
    is_opened = False
    _instance = None

    int32_max = 2147483647

    def __init__(self, n=1, offset=0, n_min=1, offset_min=0, title='Int Int Input Dialog', label='Set xrange',  parent=None):
        super(IntIntInputDialog, self).__init__(parent)
        self.setupUi(self)

        #disable resizing of the window,
        # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        self.setWindowTitle(title)
        self.label.setText(label)

        # self.dbX0.valueChanged.connect(self.dbX0_value_changed)
        # self.dbX1.valueChanged.connect(self.dbX1_value_changed)

        self.sbn.setValue(n)
        self.sbOffset.setValue(offset)

        self.sbn.setMinimum(n_min)
        self.sbOffset.setMinimum(offset_min)

        self.sbn.setMaximum(self.int32_max)
        self.sbOffset.setMaximum(self.int32_max)

        self.accepted = False

        IntIntInputDialog.is_opened = True
        IntIntInputDialog._instance = self

        self.sbn.setFocus()
        self.sbn.selectAll()

        self.show()
        self.exec()

    @staticmethod
    def get_instance():
        return IntIntInputDialog._instance

    def set_result(self):
        self.returned_range = (self.sbn.value(), self.sbOffset.value())

    def accept(self):
        self.set_result()
        self.accepted = True
        IntIntInputDialog.is_opened = False
        IntIntInputDialog._instance = None
        super(IntIntInputDialog, self).accept()

    def reject(self):
        IntIntInputDialog.is_opened = False
        IntIntInputDialog._instance = None
        super(IntIntInputDialog, self).reject()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = IntIntInputDialog()
    # Dialog.show()
    sys.exit(app.exec_())