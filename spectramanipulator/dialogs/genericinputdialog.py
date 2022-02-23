from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from ..singleton import Singleton
# import sys


class GenericInputDialog(QtWidgets.QDialog, metaclass=Singleton):

    # static variables
    is_opened = False

    def __init__(self, widget_list=None, label_text='Some descriptive text...', title='GenericInputDialog',
                 parent=None, set_result=None, flags=Qt.WindowStaysOnTopHint | Qt.MSWindowsFixedSizeDialogHint):

        super(GenericInputDialog, self).__init__(parent, flags)

        # # disable resizing of the window,
        # # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        # self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        self.set_result = set_result

        self.setWindowTitle(title)
        # self.accepted = False

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setOrientation(QtCore.Qt.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.VLayout = QVBoxLayout()
        # self.VLayout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label_text)

        self.VLayout.addWidget(self.label)

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(7)
        self.grid_layout.setContentsMargins(15, 0, 0, 0)

        if widget_list is None:
            widget_list = [("par1", QLineEdit("some  text")),
                           ("par2", QLineEdit("another  text")),
                           ("par3", QCheckBox("...."))]

        self.widget_list = []

        for i, (label, widget) in enumerate(widget_list):
            if isinstance(label, str):
                # self.label_list.append(QLabel(label))
                label = QLabel(label)
                label.setWordWrap(True)
            if widget is not None:
                self.widget_list.append(widget)
                col_span = 2 if label is None else 1
                self.grid_layout.addWidget(widget, i, 1, 1, col_span)

            if label is not None:
                col_span = 2 if widget is None else 1
                self.grid_layout.addWidget(label, i, 0, 1, col_span)

        self.VLayout.addLayout(self.grid_layout)
        self.VLayout.addWidget(self.button_box)

        self.setLayout(self.VLayout)

        self.widget_list[0].setFocus()
        if isinstance(self.widget_list[0], QLineEdit):
            self.widget_list[0].selectAll()

    def show(self):
        if self.is_opened:
            self.activateWindow()
            self.setFocus()
            return
        self.is_opened = True
        super(GenericInputDialog, self).show()

    def accept(self):
        self.set_result()
        # self.accepted = True
        self.is_opened = False
        super(GenericInputDialog, self).accept()

    def reject(self):
        self.is_opened = False
        super(GenericInputDialog, self).reject()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = GenericInputDialog()
    Dialog.show()
    sys.exit(app.exec_())