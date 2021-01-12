

from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal


class CheckBoxRC(QCheckBox):
    """Classical checkbox that allows right click which toggles the checkbox the same as left click,
    if checkbox was right clicked, the attribute self.right_button_pressed will be set to True."""

    def __init__(self, text: str = '', parent=None):
        super(CheckBoxRC, self).__init__(text, parent)

        self.right_button_pressed = False

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            # print('left button clicked')
            self.right_button_pressed = False
            super(CheckBoxRC, self).mousePressEvent(e)
        elif e.button() == Qt.RightButton:
            # print('right button clicked')
            self.right_button_pressed = True
            self.setChecked(not self.isChecked())
            # self.right_button_clicked.emit(checked)


