
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from spectramanipulator.singleton import PersistentOKCancelDialog
from typing import Callable
from PyQt5.QtWidgets import QLabel, QSpinBox, QGridLayout, QVBoxLayout, QLineEdit, QCheckBox


class RenameDialog(PersistentOKCancelDialog):

    int32_max = 2147483647

    def __init__(self, accepted_func: Callable, expression='', offset=0, c_mult_factor=1,
                 last_rename_take_name_from_list=False, parent=None):
        super(RenameDialog, self).__init__(accepted_func, parent)
        self.setWindowTitle("Rename Items")

        self.result = (expression, offset, c_mult_factor)
        self.list = last_rename_take_name_from_list

        self.description = QLabel("Renames selected items. {:2d} - integer counter, {:2f} - float counter, {:2g} - significant digits counter. for more see https://pyformat.info/. For original name slicing, use {start_idx:end_idx}, same from python slicing rules. Eg. expression is '{:02d}: t = {:} us', current name is '167' and current counter is 16. Resulting name will be '16: t = 167 us'. Current counter (integer counter starts with counter offset value) for each item is multiplied by counter mult. factor.")
        self.description.setWordWrap(True)

        self.leExpression = QLineEdit()
        self.sbOffset = QSpinBox()
        self.leCounterMulFactor = QLineEdit()

        self.leExpression.setText(expression)
        self.leCounterMulFactor.setText(str(c_mult_factor))

        self.sbOffset.setValue(offset)
        self.sbOffset.setMinimum(0)
        self.sbOffset.setMaximum(self.int32_max)

        self.cbTakeNamesFromList = QCheckBox("Take names from list (separate values by comma):")
        self.cbTakeNamesFromList.setChecked(last_rename_take_name_from_list)

        self.leList = QLineEdit()

        self.grid_layout = QGridLayout()

        self.grid_layout.addWidget(QLabel('Expression:'), 0, 0, 1, 1)
        self.grid_layout.addWidget(QLabel('Counter offset:'), 1, 0, 1, 1)
        self.grid_layout.addWidget(QLabel('Counter mult. factor:'), 2, 0, 1, 1)
        self.grid_layout.addWidget(self.leExpression, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.sbOffset, 1, 1, 1, 1)
        self.grid_layout.addWidget(self.leCounterMulFactor, 2, 1, 1, 1)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.description)
        self.layout.addLayout(self.grid_layout)
        self.layout.addWidget(self.cbTakeNamesFromList)
        self.layout.addWidget(self.leList)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)

        self.leExpression.setFocus()
        self.leExpression.selectAll()

        self.cbTakeNamesFromList.stateChanged.connect(self.cbTakeNamesFromList_check_changed)

        # perform change
        self.cbTakeNamesFromList_check_changed()

    # def set_result(self):
    #     if not self.cbTakeNamesFromList.isChecked():
    #         self.result = (self.leExpression.text(), self.sbOffset.value(), self.leCounterMulFactor.text())
    #     else:
    #         self.list = self.leList.text()

    def cbTakeNamesFromList_check_changed(self):
        is_checked = self.cbTakeNamesFromList.isChecked()
        self.sbOffset.setEnabled(not is_checked)
        self.leExpression.setEnabled(not is_checked)
        self.leList.setEnabled(is_checked)
        self.leCounterMulFactor.setEnabled(not is_checked)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = RenameDialog(None)
    Dialog.show()
    sys.exit(app.exec_())

