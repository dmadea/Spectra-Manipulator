# from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QMessageBox, QDialogButtonBox, QVBoxLayout, QLabel, QGridLayout
from spectramanipulator.singleton import InputWidget
from .mylineedit import MyLineEdit
from PyQt5.QtCore import Qt

from ..plotwidget import PlotWidget
# import pyqtgraph as pg
from typing import Callable


class RangeWidget(InputWidget):

    def __init__(self, dock_widget, accepted_func: Callable = None,
                 label_text='Set xrange:', title='SetRangeDialog', parent=None):

        super(RangeWidget, self).__init__(dock_widget, accepted_func, title, parent)

        self.pw = PlotWidget()

        self.le_x0 = MyLineEdit()
        self.le_x1 = MyLineEdit()

        self.VLayout = QVBoxLayout()
        # self.VLayout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label_text)
        self.label.setWordWrap(True)

        self.VLayout.addWidget(self.label)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(7)
        self.grid_layout.setContentsMargins(15, 0, 0, 0)

        self.grid_layout.addWidget(QLabel('x0:'), 0, 0)
        self.grid_layout.addWidget(self.le_x0, 0, 1)
        self.grid_layout.addWidget(QLabel('x1:'), 1, 0)
        self.grid_layout.addWidget(self.le_x1, 1, 1)

        self.VLayout.addItem(self.grid_layout)
        self.VLayout.addWidget(self.button_box)

        self.setLayout(self.VLayout)

        self.lr = self.pw.add_linear_region(z_value=1e8)
        self.lr.sigRegionChanged.connect(lambda: self.update_values())

        self.le_x0.focus_lost.connect(self.update_region)
        self.le_x1.focus_lost.connect(self.update_region)
        self.le_x0.returnPressed.connect(self.update_region)
        self.le_x1.returnPressed.connect(self.update_region)

        self.lr.sigRegionChanged.connect(self.update_values)

        self.update_values()

        # we have to update region, otherwise there would be some bug in with manually moving the region
        self.update_region()
        self.returned_range = None

        # set focus
        self.le_x0.setFocus(Qt.TabFocusReason)
        self.le_x0.selectAll()

    def update_values(self):
        x0, x1 = self.lr.getRegion()
        self.le_x0.setText("{:.4g}".format(x0))
        self.le_x1.setText("{:.4g}".format(x1))

    def update_region(self):
        try:
            x0, x1 = float(self.le_x0.text()), float(self.le_x1.text())
            if x0 <= x1:
                self.lr.setRegion((x0, x1))
        except ValueError:
            pass

    def accept(self):
        try:
            self.returned_range = [float(self.le_x0.text()), float(self.le_x1.text())]

            # swap range if it is reversed
            if self.returned_range[0] > self.returned_range[1]:
                temp = self.returned_range[0]
                self.returned_range[0] = self.returned_range[1]
                self.returned_range[1] = temp

        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid format of the range.", QMessageBox.Ok)
            return

        self.pw.remove_linear_region()
        super(RangeWidget, self).accept()

    def reject(self):
        self.pw.remove_linear_region()
        super(RangeWidget, self).reject()


# if __name__ == "__main__":
    # import sys
    # from PyQt5.QtWidgets import QApplication
    #
    # app = QApplication(sys.argv)
    # Dialog = RangeWidget()
    # Dialog.show()
    # sys.exit(app.exec_())
