
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtCore import pyqtSignal


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

    # def accept(self):
    #     self.accepted_signal.emit()
    #     super(PersistentDialog, self).accept()


