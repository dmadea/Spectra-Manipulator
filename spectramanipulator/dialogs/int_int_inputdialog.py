
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QSpinBox, QGridLayout, QVBoxLayout
# from PyQt5.QtCore import Qt
from spectramanipulator.singleton import PersistentOKCancelDialog
from typing import Callable


class IntIntInputDialog(PersistentOKCancelDialog):

    int32_max = 2147483647

    def __init__(self, accepted_func: Callable, n=1, offset=0, n_min=1, offset_min=0,
                 title='Int Int Input Dialog', label='Set xrange', parent=None):
        super(IntIntInputDialog, self).__init__(accepted_func, parent)

        self.setWindowTitle(title)
        self.label = QLabel(label)

        self.sbn = QSpinBox()
        self.sbOffset = QSpinBox()

        self.sbn.setValue(n)
        self.sbOffset.setValue(offset)

        self.sbn.setMinimum(n_min)
        self.sbOffset.setMinimum(offset_min)

        self.sbn.setMaximum(self.int32_max)
        self.sbOffset.setMaximum(self.int32_max)

        self.grid_layout = QGridLayout()

        self.grid_layout.addWidget(QLabel('n'), 0, 0, 1, 1)
        self.grid_layout.addWidget(QLabel('shift'), 1, 0, 1, 1)
        self.grid_layout.addWidget(self.sbn, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.sbOffset, 1, 1, 1, 1)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addLayout(self.grid_layout)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)

        self.sbn.setFocus()
        self.sbn.selectAll()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = IntIntInputDialog(None)
    Dialog.show()
    sys.exit(app.exec_())

