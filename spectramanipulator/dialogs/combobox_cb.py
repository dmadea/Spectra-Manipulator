from PyQt5.QtWidgets import QComboBox, QApplication
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt, pyqtSignal

from ..logger import Logger


class ComboBoxCB(QComboBox):

    check_changed = pyqtSignal(list)  # list of abbreviations of checked items

    def __init__(self, parent=None, items: dict = None):
        """items is dictionary that uses short names as keys and longer description as values"""
        super(ComboBoxCB, self).__init__(parent)

        self.setEditable(True)

        self.cb_items = []  # list of checkboxes
        self.items = {} if items is None else items
        self.model = QStandardItemModel(1, 1)
        self.setModel(self.model)
        self.model.dataChanged.connect(self.data_changed)

        self.setting_check_state = False

    def set_check_state(self, i, checked=True):
        if i >= len(self.cb_items):
            return

        self.setting_check_state = True

        self.cb_items[i].setData(Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole)

        self.setting_check_state = False

    def set_data(self, items: dict):
        self.cb_items = []
        self.items = items
        self.model.setRowCount(len(self.items))

        for i, (key, value) in enumerate(self.items.items()):
            item = QStandardItem(f'{key}: {value}')

            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.Unchecked, Qt.CheckStateRole)

            self.cb_items.append(item)

            self.model.setItem(i, 0, item)

        self.set_text("")

    def update_text(self):
        self.data_changed(None, None, None, None)

    def data_changed(self, index1, index2, roles, p_int=None):
        if self.setting_check_state:
            return

        text = ''
        if len(self.cb_items) == len(self.items):
            key_list = [key for i, (key, value) in enumerate(self.items.items()) if
                        self.cb_items[i].checkState() == Qt.Checked]
            text = ', '.join(key_list)
            self.check_changed.emit(key_list)

        self.set_text(text)
        Logger.debug('data_changed called')

    def set_text(self, text):
        l_edit = self.lineEdit()
        l_edit.setText(text)
        l_edit.setReadOnly(True)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    cb = ComboBoxCB(None, ['a', 'b', 'c'])
    cb.show()
    sys.exit(app.exec_())
