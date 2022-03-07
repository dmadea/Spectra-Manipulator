
from PyQt5.QtWidgets import QDialog, QWidget, QDialogButtonBox
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtCore import Qt
# import types
# from six import with_metaclass
# import logging
from typing import Callable

# from https://forum.qt.io/topic/88531/singleton-in-python-with-qobject/2

try:
    from PyQt5.QtCore import pyqtWrapperType
except ImportError:
    from sip import wrappertype as pyqtWrapperType


# also from # https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python?page=1&tab=votes#tab-top
class Singleton(pyqtWrapperType, type):
    """
        Singleton metaclass.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances or cls._instances[cls] is None:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def remove_instance(cls):
        """
        Sets the instance of a class to None.
        """
        cls._instances[cls] = None


class InputWidget(QWidget):

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
        # logging.warning('InputWidget closeEvent called.')
        self._is_visible = False
        self.dock_widget.setVisible(False)
        super(InputWidget, self).closeEvent(a0)


# the instance is removed after the dialog is closed
class PersistentDialog(QDialog, metaclass=Singleton):
    """
    Singleton-type class that implement persistent dialog. After it is instantiated, it stores
    its instance. After its closure, the instance is removed.
    """
    _is_opened = False

    def show(self):
        if self._is_opened:
            self.activateWindow()
            self.setFocus()
            return
        self._is_opened = True
        super(PersistentDialog, self).show()

    def reject(self) -> None:
        self._is_opened = False  # need to do it here
        self.__class__.remove_instance()  # set the instance to None
        super(PersistentDialog, self).reject()  # does not call the closeevent...

    def accept(self) -> None:
        self._is_opened = False
        self.__class__.remove_instance()  # set the instance to None
        super(PersistentDialog, self).accept()


class PersistentOKCancelDialog(PersistentDialog):

    def __init__(self, accepted_func: Callable, parent=None, **kwargs):
        super(PersistentOKCancelDialog, self).__init__(parent=parent, **kwargs)
        self.accepted_func = accepted_func

        self.button_box = QDialogButtonBox()
        self.button_box.setOrientation(Qt.Horizontal)
        self.button_box.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        self.button_box.accepted.connect(self.accept)  # OK button
        self.button_box.rejected.connect(self.reject)  # Cancel button

    def accept(self) -> None:
        self.accepted_func()
        super(PersistentOKCancelDialog, self).accept()


class TestClass(object, metaclass=Singleton):

    def __init__(self, b):
        self.b = b


if __name__ == '__main__':
    # pass
    t = TestClass('asd')
    b = TestClass(4865)
    # pD = PersistentDialog()

    print(t == b)
    # t = TestClass(5)
    # t = TestClass(5)
    # t = TestClass(5)

