from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QLineEdit, QMessageBox
from PyQt5.QtCore import pyqtSignal
from dialogs.genericinputdialog import GenericInputDialog

from plotwidget import PlotWidget
import pyqtgraph as pg


class MyLineEdit(QLineEdit):
    focus_lost = pyqtSignal()

    def focusOutEvent(self, e):
        self.focus_lost.emit()
        # a= QFocusEvent()
        # print(self.text(), "lost focus")
        super(MyLineEdit, self).focusOutEvent(e)


class RangeDialog(GenericInputDialog):

    def __init__(self, label_text='Set xrange', title='SetRangeDialog', parent=None):

        self.le_x0 = MyLineEdit()
        self.le_x1 = MyLineEdit()

        super(RangeDialog, self).__init__([('x0', self.le_x0), ('x1', self.le_x1)],
                                          label_text=label_text,
                                          title=title, parent=parent)

        f = 0.87
        # x0, x1 = x0_value, x1_value
        x0, x1 = PlotWidget.instance.getViewBox().viewRange()[0]
        xy_dif = x1 - x0
        self.lr = pg.LinearRegionItem([x0 + (1 - f) * xy_dif, x0 + f * xy_dif],
                                      brush=QBrush(QColor(0, 255, 0, 20)))

        self.lr.setZValue(-10)
        PlotWidget.instance.addItem(self.lr)

        self.le_x0.focus_lost.connect(self.update_region)
        self.le_x1.focus_lost.connect(self.update_region)

        self.lr.sigRegionChanged.connect(self.update_values)
        self.update_values()
        # we have to update region, otherwide there would be some bug in with manually mooving the region
        self.update_region()

        self.returned_range = None

        self.show()
        self.exec()
        # sys.exit(self.exec_())

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

            # swap range if it is revesed
            if self.returned_range[0] > self.returned_range[1]:
                temp = self.returned_range[0]
                self.returned_range[0] = self.returned_range[1]
                self.returned_range[1] = temp

        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid format of the range.", QMessageBox.Ok)
            return

        PlotWidget.instance.removeItem(self.lr)
        # del self.lr
        super(RangeDialog, self).accept()

    def reject(self):
        PlotWidget.instance.removeItem(self.lr)
        # del self.lr
        super(RangeDialog, self).reject()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    Dialog = RangeDialog()
    Dialog.show()
    sys.exit(app.exec_())
