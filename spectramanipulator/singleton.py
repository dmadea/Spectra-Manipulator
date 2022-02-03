
from PyQt5.QtWidgets import QDialog, QWidget
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtCore import Qt

import logging


class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

# as meta class
# class Singleton(type):
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]


class InputWidget(QWidget, Singleton):

    _is_visible = False

    def __init__(self, dock_widget, title='title', parent=None):
        super(InputWidget, self).__init__(parent=parent)

        self.dock_widget = dock_widget

        self.dock_widget.parent().resizeDocks([self.dock_widget], [250], Qt.Vertical)
        self.dock_widget.titleBarWidget().setText(title)
        self.dock_widget.setWidget(self)
        self.dock_widget.setVisible(True)

        self._is_visible = True

    def closeEvent(self, a0: QCloseEvent):
        logging.warning('InputWidget closeEvent called.')
        self._is_visible = False
        self.dock_widget.setVisible(False)
        super(InputWidget, self).closeEvent(a0)


class PersistentDialog(Singleton, QDialog):

    _is_opened = False

    def show(self):
        if self._is_opened:
            self.activateWindow()
            self.setFocus()
            return
        self._is_opened = True
        super(PersistentDialog, self).show()

    def closeEvent(self, a0: QCloseEvent):
        # logging.info('Settings closeEvent.')
        self._is_opened = False
        super(PersistentDialog, self).closeEvent(a0)

