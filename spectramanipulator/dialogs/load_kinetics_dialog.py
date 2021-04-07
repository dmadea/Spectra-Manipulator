
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QListView, QAbstractItemView, QTreeView, QListWidget

from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from spectramanipulator.dialogs.gui_load_kinetics import Ui_Dialog
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


class LoadKineticsDialog(QtWidgets.QDialog, Ui_Dialog):

    # static variables
    is_opened = False
    _instance = None

    int32_max = 2147483647

    def __init__(self, parent=None):
        super(LoadKineticsDialog, self).__init__(parent)
        self.setupUi(self)

        self.lwFolders = MyQListWidget(self)

        #disable resizing of the window,
        # help from https://stackoverflow.com/questions/16673074/in-qt-c-how-can-i-fully-disable-resizing-a-window-including-the-resize-icon-w
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        self.setWindowTitle("Batch Load Kinetics")

        self.cbKineticsMeasuredByEach.toggled.connect(self.cbKineticsMeasuredByEach_toggled)
        self.cbBCorr.toggled.connect(self.cbBCorr_toggled)
        self.cbCut.toggled.connect(self.cbCut_toggled)
        self.btnChooseDirs.clicked.connect(self.btnChooseDirs_clicked)

        self.cbKineticsMeasuredByEach_toggled()
        self.cbBCorr_toggled()
        self.cbCut_toggled()

        self.lwGridLayout.addWidget(self.lwFolders, 0, 0, 0, 0)

        self.accepted = False

        LoadKineticsDialog.is_opened = True
        LoadKineticsDialog._instance = self

        self.show()
        self.exec()

    def lwFolders_keyPressEvent(self, e):
        print(e)

    @staticmethod
    def get_instance():
        return LoadKineticsDialog._instance

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else 0

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        print(e)

    def btnChooseDirs_clicked(self):
        # https://stackoverflow.com/questions/38252419/how-to-get-qfiledialog-to-select-and-return-multiple-folders
        # just copied :)
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
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

    def set_result(self):
        pass

        # if self.is_renaming_by_expression:
        #     self.result = (self.leExpression.text(), self.sbOffset.value(), self.leCounterMulFactor.text())
        # else:
        #     self.list = self.leList.text()

    # def cbTakeNamesFromList_check_changed(self):
        # if self.cbTakeNamesFromList.checkState() == Qt.Checked:
        #     self.sbOffset.setEnabled(False)
        #     self.leExpression.setEnabled(False)
        #     self.leList.setEnabled(True)
        #     self.leCounterMulFactor.setEnabled(False)
        #     self.is_renaming_by_expression = False
        # else:
        #     self.sbOffset.setEnabled(True)
        #     self.leExpression.setEnabled(True)
        #     self.leList.setEnabled(False)
        #     self.leCounterMulFactor.setEnabled(True)
        #     self.is_renaming_by_expression = True

    def accept(self):
        self.set_result()
        self.accepted = True
        LoadKineticsDialog.is_opened = False
        LoadKineticsDialog._instance = None
        super(LoadKineticsDialog, self).accept()

    def reject(self):
        LoadKineticsDialog.is_opened = False
        LoadKineticsDialog._instance = None
        super(LoadKineticsDialog, self).reject()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = LoadKineticsDialog()
    # Dialog.show()
    sys.exit(app.exec_())


