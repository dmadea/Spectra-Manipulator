
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QLabel, QDialogButtonBox, QGridLayout, QLineEdit, QComboBox, QVBoxLayout

from PyQt5.QtCore import Qt
from spectramanipulator.singleton import PersistentOKCancelDialog

from webbrowser import open_new_tab
from typing import Callable


class InterpolateDialog(PersistentOKCancelDialog):

    def __init__(self, accepted_func: Callable, parent=None):
        super(InterpolateDialog, self).__init__(accepted_func, parent)

        self.setWindowTitle("Interpolate")

        self.spacing = 1
        self.selected_kind = 'linear'

        self.kinds = [
            {'kind': 'linear', 'name': 'Linear'},
            {'kind': 'nearest', 'name': 'Nearest'},
            {'kind': 'previous', 'name': 'Previous'},
            {'kind': 'next', 'name': 'Next'},
            {'kind': 'zero', 'name': 'Spline (zero order)'},
            {'kind': 'slinear', 'name': 'Spline (1st order)'},
            {'kind': 'quadratic', 'name': 'Spline (2nd order)'},
            {'kind': 'cubic', 'name': 'Spline (3rd order)'},
        ]

        self.description_label = QLabel(
            "<html><head/><body><p>Set the spacing of x values and kind of interpolation. For documentation, see <a href=\"https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d\"><span style=\" text-decoration: underline; color:#0000ff;\">scipy.interpolate.interp1d.</span></a> Eg. for resampling the unevenly spaced spectral data by 1 nm, set spacing to 1.</p></body></html>")
        self.description_label.setWordWrap(True)

        self.grid_layout = QGridLayout()

        self.leSpacing = QLineEdit(str(self.spacing))
        self.cbInterpolation = QComboBox()
        self.cbInterpolation.addItems(map(lambda d: d['name'], self.kinds))

        self.grid_layout.addWidget(QLabel('Spacing'), 0, 0, 1, 1)
        self.grid_layout.addWidget(QLabel('Interpolation'), 1, 0, 1, 1)
        self.grid_layout.addWidget(self.leSpacing, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.cbInterpolation, 1, 1, 1, 1)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.description_label)
        self.layout.addLayout(self.grid_layout)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)

        self.description_label.linkActivated.connect(lambda: open_new_tab('https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d'))

    def accept(self):
        try:
            self.spacing = float(self.leSpacing.text().replace(',', '.').strip())
        except ValueError:
            QMessageBox.warning(self, 'Error', "Cannot parse spacing value into float, please correct it.", QMessageBox.Ok)
            return

        self.selected_kind = self.kinds[self.cbInterpolation.currentIndex()]['kind']
        super(InterpolateDialog, self).accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = InterpolateDialog(None)
    Dialog.show()
    sys.exit(app.exec_())
