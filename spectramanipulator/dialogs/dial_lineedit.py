
from PyQt5.QtWidgets import QLineEdit, QWidget, QHBoxLayout, QApplication, QToolButton
from PyQt5.QtGui import QRegion, QMouseEvent, QWheelEvent
from PyQt5.QtCore import Qt, pyqtSignal

import numpy as np


def decor_t2f(func):
    def text2float():
        try:
            val = float(func())
        except ValueError:
            return
        return val
    return text2float


class DialLineEdit(QHBoxLayout):

    # check_changed = pyqtSignal(list)  # list of abbreviations of checked items
    simulation_requested = pyqtSignal()

    def __init__(self, text: str = '', lb_val=lambda: -10, ub_val=lambda: np.inf, parent=None):
        """items is dictionary that uses short names as keys and longer description as values"""
        super(DialLineEdit, self).__init__(parent)

        self.lb_val = decor_t2f(lb_val)  # lower bound func
        self.ub_val = decor_t2f(ub_val)  # upper bound func

        self.le = _DialLineEdit(text, self)
        self.dial = RoundButton(self.le, self)
        self.addWidget(self.le)
        self.addWidget(self.dial)

    def setVisible(self, visible):
        self.le.setVisible(visible)
        self.dial.setVisible(visible)

    def setEnabled(self, enabled: bool):
        self.le.setEnabled(enabled)
        self.dial.setEnabled(enabled)
        super(DialLineEdit, self).setEnabled(enabled)

    def __getattr__(self, item):  # redirect to main QLineEdit
        return self.le.__getattribute__(item)


class _DialLineEdit(QLineEdit):

    def __init__(self, text: str = '', parent=None):
        super(_DialLineEdit, self).__init__(text, None)
        self.lb_val = parent.lb_val  # lower bound func
        self.ub_val = parent.ub_val  # upper bound func
        self.simulation_requested = parent.simulation_requested
        self.returnPressed.connect(lambda: self.simulation_requested.emit())  # simulate when enter was pressed

    def wheelEvent(self, e: QWheelEvent):
        super(_DialLineEdit, self).wheelEvent(e)
        turns = e.angleDelta().y() // 120

        try:
            val = float(self.text())
        except ValueError:
            return

        lb = self.lb_val()
        ub = self.ub_val()

        if lb is None or ub is None:
            return

        val += val * turns * np.sign(val) / 10  # increase/decrease in 10th of original value

        if val < lb:
            val = lb

        if val > ub:
            val = ub

        self.setText(f'{val:.4g}')
        self.simulation_requested.emit()


class RoundButton(QToolButton):
    def __init__(self, lineedit, parent=None):
        super(RoundButton, self).__init__(None)

        self.lb_val = parent.lb_val  # lower bound func
        self.ub_val = parent.ub_val  # upper bound func
        self.simulation_requested = parent.simulation_requested

        # self.left_btn_pressed = False
        self.init_val = None
        self.initial_y_pos = 0
        self.le = lineedit

        stylesheet = """
        RoundButton {
            background-color : darkgreen;
        }
        
        RoundButton::hover {
            background-color : darkred;
        }
        
        RoundButton::pressed {
            background-color : red;
        }
        """
        self.setStyleSheet(stylesheet)

    def mouseMoveEvent(self, e: QMouseEvent):
        super(RoundButton, self).mouseMoveEvent(e)

        if self.init_val is None:
            return

        lb = self.lb_val()
        ub = self.ub_val()

        if lb is None or ub is None:
            return

        dy = self.initial_y_pos - e.y()
        val = self.init_val * (1 + np.sign(self.init_val) * dy * 1e-2)

        if val < lb:
            val = lb

        if val > ub:
            val = ub

        self.le.setText(f'{val:.4g}')
        self.simulation_requested.emit()

    def mousePressEvent(self, e: QMouseEvent):
        try:
            val = float(self.le.text())
        except ValueError:
            return

        self.init_val = val
        self.initial_y_pos = e.y()
        super(RoundButton, self).mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent):
        self.init_val = None
        super(RoundButton, self).mouseReleaseEvent(e)

    def resizeEvent(self, e):
        r = self.rect()
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        x += w // 4
        y += h // 4
        w //= 2
        h //= 2
        self.setMask(QRegion(x, y, w, h, QRegion.Ellipse))
        super(RoundButton, self).resizeEvent(e)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    cb = DialLineEdit('10')
    cb.show()
    sys.exit(app.exec_())
