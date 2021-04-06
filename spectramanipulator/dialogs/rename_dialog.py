
from PyQt5 import QtWidgets

from PyQt5.QtCore import Qt
from .gui_rename_dialog import Ui_Dialog


class RenameDialog(QtWidgets.QDialog, Ui_Dialog):

    # static variables
    is_opened = False
    _instance = None

    int32_max = 2147483647

    def __init__(self, expression='', offset=0, c_mult_facotr=1,
                 last_rename_take_name_from_list=False,  parent=None):
        super(RenameDialog, self).__init__(parent)
        self.setupUi(self)
        self.result = (expression, offset, c_mult_facotr)
        self.list = last_rename_take_name_from_list

        #disable resizing of the window,
        # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        self.setWindowTitle("Rename Items")

        self.leExpression.setText(expression)
        self.leCounterMulFactor.setText(str(c_mult_facotr))

        self.sbOffset.setValue(offset)
        self.sbOffset.setMinimum(0)
        self.sbOffset.setMaximum(self.int32_max)

        self.cbTakeNamesFromList.setCheckState(self.check_state(last_rename_take_name_from_list))

        self.accepted = False

        RenameDialog.is_opened = True
        RenameDialog._instance = self

        self.leExpression.setFocus()
        self.leExpression.selectAll()

        self.cbTakeNamesFromList.stateChanged.connect(self.cbTakeNamesFromList_check_changed)

        # perform change
        self.cbTakeNamesFromList_check_changed()

        self.show()
        self.exec()

    @staticmethod
    def get_instance():
        return RenameDialog._instance

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else 0

    def set_result(self):

        if self.is_renaming_by_expression:
            self.result = (self.leExpression.text(), self.sbOffset.value(), self.leCounterMulFactor.text())
        else:
            self.list = self.leList.text()

    def cbTakeNamesFromList_check_changed(self):
        if self.cbTakeNamesFromList.checkState() == Qt.Checked:
            self.sbOffset.setEnabled(False)
            self.leExpression.setEnabled(False)
            self.leList.setEnabled(True)
            self.is_renaming_by_expression = False
        else:
            self.sbOffset.setEnabled(True)
            self.leExpression.setEnabled(True)
            self.leList.setEnabled(False)
            self.is_renaming_by_expression = True

    def accept(self):
        self.set_result()
        self.accepted = True
        RenameDialog.is_opened = False
        RenameDialog._instance = None
        super(RenameDialog, self).accept()

    def reject(self):
        RenameDialog.is_opened = False
        RenameDialog._instance = None
        super(RenameDialog, self).reject()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = RenameDialog()
    # Dialog.show()
    sys.exit(app.exec_())