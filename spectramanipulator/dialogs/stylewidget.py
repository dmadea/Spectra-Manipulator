from PyQt5 import QtCore, QtGui, QtWidgets

from .stylewidget_gui import Ui_Form
from PyQt5.QtWidgets import QColorDialog
from PyQt5.QtGui import QColor
# from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox
# import numpy as np
#
# from ..spectrum import Spectrum
#
# import pyqtgraph as pg


from PyQt5.QtCore import Qt
from ..settings.settings import Settings


class StyleWidget(QtWidgets.QWidget, Ui_Form):
    # static variables
    is_opened = False
    _instance = None

    def __init__(self, dock_widget, accepted_func, selected_item, parent=None):
        super(StyleWidget, self).__init__(parent)

        self.setupUi(self)

        self.dock_widget = dock_widget
        self.accepted_func = accepted_func
        # self.selected_item = selected_item

        self.color = None
        self.sym_brush_color = None
        self.sym_fill_color = None

        self.btnOK.clicked.connect(self.accept)
        self.btnApply.clicked.connect(self.applied)
        self.btnCancel.clicked.connect(self.reject)
        self.btnSetDefauls.clicked.connect(self.set_defaults)
        self.cbColor.toggled.connect(self.cbColor_checked_changed)
        self.cbLineWidth.toggled.connect(self.cbLineWidth_checked_changed)
        self.cbLineType.toggled.connect(self.cbLineType_checked_changed)
        self.btnColor.clicked.connect(self.pick_color_clicked)
        self.btnSymBrushColor.clicked.connect(self.pick_color_brush_clicked)
        self.btnSymFillColor.clicked.connect(self.pick_color_fill_clicked)

        self.cbSymBrushDefault.toggled.connect(self.cbSymBrushDefault_checked_changed)
        self.cbSymFillDefault.toggled.connect(self.cbSymFillDefault_checked_changed)

        self.line_types = [
            {'name': 'Solid line', 'index': Qt.SolidLine},
            {'name': 'Dashed line', 'index': Qt.DashLine},
            {'name': 'Dotted line', 'index': Qt.DotLine},
            {'name': 'Dash-dotted line', 'index': Qt.DashDotLine},
            {'name': 'Dash-dot-dotted line', 'index': Qt.DashDotDotLine},
            {'name': 'No line', 'index': Qt.NoPen}
        ]

        self.symbol_types = [
            {'name': 'No symbol', 'sym': None},
            {'name': '\u25CF', 'sym': 'o'},
            {'name': '\u25BC', 'sym': 't'},
            {'name': '\u25B2', 'sym': 't1'},
            {'name': '\u25BA', 'sym': 't2'},
            {'name': '\u25C4', 'sym': 't3'},
            {'name': '\u25A0', 'sym': 's'},
            {'name': '\u2B1F', 'sym': 'p'},
            {'name': '\u2B22', 'sym': 'h'},
            {'name': '\u2605', 'sym': 'star'},
            {'name': '+', 'sym': '+'},
            {'name': '\u2666', 'sym': 'd'}
        ]

        self.combLineType.addItems(map(lambda m: m['name'], self.line_types))
        self.combSymbol.addItems(map(lambda m: m['name'], self.symbol_types))

        # self.sbAlpha.setValue(255)
        # self.sbSymBrushAlpha.setValue(255)
        # self.sbSymFillAlpha.setValue(255)

        self.dsbLineWidth.setValue(Settings.line_width)
        self.combLineType.setCurrentIndex(0)
        self.combSymbol.setCurrentIndex(0)

        if not hasattr(selected_item, 'line_alpha'):
            setattr(selected_item, 'line_alpha', 255)

        if selected_item.color is None:
            self.cbColor.setChecked(True)
        else:
            self.color = QColor(*selected_item.color) if isinstance(selected_item.color, (tuple, list)) else QColor(
                selected_item.color)

            self.btnColor.setStyleSheet("QWidget {background-color: %s}" % self.color.name())

        self.sbAlpha.setValue(selected_item.line_alpha)

        if selected_item.line_width is None:
            self.cbLineWidth.setChecked(True)
        else:
            self.dsbLineWidth.setValue(float(selected_item.line_width))

        if selected_item.line_type is None:
            self.cbLineType.setChecked(True)
        else:
            curr_type = list(filter(lambda t: t['index'] == selected_item.line_type, self.line_types))
            self.combLineType.setCurrentIndex(self.line_types.index(curr_type[0]))

        if not hasattr(selected_item, 'symbol'):
            setattr(selected_item, 'symbol', None)

        if not hasattr(selected_item, 'symbol_brush'):
            setattr(selected_item, 'symbol_brush', None)

        if not hasattr(selected_item, 'symbol_fill'):
            setattr(selected_item, 'symbol_fill', None)

        if not hasattr(selected_item, 'symbol_size'):
            setattr(selected_item, 'symbol_size', 8)

        if not hasattr(selected_item, 'sym_brush_alpha'):
            setattr(selected_item, 'sym_brush_alpha', 255)

        if not hasattr(selected_item, 'sym_fill_alpha'):
            setattr(selected_item, 'sym_fill_alpha', 255)

        if selected_item.symbol is None:
            self.combSymbol.setCurrentIndex(0)
        else:
            curr_sym_type = list(filter(lambda t: t['sym'] == selected_item.symbol, self.symbol_types))
            self.combSymbol.setCurrentIndex(self.symbol_types.index(curr_sym_type[0]))

        if selected_item.symbol_brush is None:
            self.cbSymBrushDefault.setChecked(True)
        else:
            self.sym_brush_color = QColor(*selected_item.color) if isinstance(selected_item.symbol_brush, (tuple, list)) else QColor(
                selected_item.symbol_brush)

            self.btnSymBrushColor.setStyleSheet("QWidget {background-color: %s}" % self.sym_brush_color.name())
            self.sbSymBrushAlpha.setValue(self.sym_brush_color.alpha())

        if selected_item.symbol_fill is None:
            self.cbSymFillDefault.setChecked(True)
        else:
            self.sym_fill_color = QColor(*selected_item.color) if isinstance(selected_item.symbol_fill, (tuple, list)) else QColor(
                selected_item.symbol_fill)

            self.btnSymFillColor.setStyleSheet("QWidget {background-color: %s}" % self.sym_fill_color.name())
            self.sbSymFillAlpha.setValue(self.sym_fill_color.alpha())

        self.sbSymBrushAlpha.setValue(selected_item.sym_brush_alpha)
        self.sbSymFillAlpha.setValue(selected_item.sym_fill_alpha)
        self.dsbSymSize.setValue(float(selected_item.symbol_size))

        if not hasattr(selected_item, 'plot_legend'):
            setattr(selected_item, 'plot_legend', True)

        self.cbPlotLegend.setChecked(selected_item.plot_legend)

        self.accepted = False
        StyleWidget.is_opened = True
        StyleWidget._instance = self

        self.dock_widget.parent().resizeDocks([self.dock_widget], [150], Qt.Vertical)
        self.dock_widget.titleBarWidget().setText("Style settings")
        self.dock_widget.setWidget(self)
        self.dock_widget.setVisible(True)

    def cbSymBrushDefault_checked_changed(self):
        checked = self.cbSymBrushDefault.isChecked()
        self.btnSymBrushColor.setEnabled(not checked)

    def cbSymFillDefault_checked_changed(self):
        checked = self.cbSymFillDefault.isChecked()
        self.btnSymFillColor.setEnabled(not checked)

    def cbColor_checked_changed(self):
        checked = self.cbColor.isChecked()
        self.btnColor.setEnabled(not checked)
        # self.sbAlpha.setEnabled(not checked)

    def cbLineWidth_checked_changed(self):
        checked = self.cbLineWidth.isChecked()
        self.dsbLineWidth.setEnabled(not checked)

    def cbLineType_checked_changed(self):
        checked = self.cbLineType.isChecked()
        self.combLineType.setEnabled(not checked)

    def pick_color_clicked(self):
        color = QColorDialog.getColor(self.color if self.color else QColor('white'))
        if color.isValid():
            self.btnColor.setStyleSheet("QWidget { background-color: %s}" % color.name())
            self.color = color

    def pick_color_brush_clicked(self):
        color = QColorDialog.getColor(self.sym_brush_color if self.sym_brush_color else QColor('white'))
        if color.isValid():
            self.btnSymBrushColor.setStyleSheet("QWidget { background-color: %s}" % color.name())
            self.sym_brush_color = color

    def pick_color_fill_clicked(self):
        color = QColorDialog.getColor(self.sym_fill_color if self.sym_fill_color else QColor('white'))
        if color.isValid():
            self.btnSymFillColor.setStyleSheet("QWidget { background-color: %s}" % color.name())
            self.sym_fill_color = color

    def set_defaults(self):
        self.cbColor.setChecked(True)
        self.cbLineType.setChecked(True)
        self.cbLineWidth.setChecked(True)
        self.cbSymBrushDefault.setChecked(True)
        self.cbSymFillDefault.setChecked(True)
        self.combLineType.setCurrentIndex(0)
        self.combSymbol.setCurrentIndex(0)

    def applied(self):
        self.accepted_func()

    def accept(self):
        self.accepted = True
        StyleWidget.is_opened = False
        StyleWidget._instance = None
        self.dock_widget.setVisible(False)
        self.accepted_func()

    def reject(self):
        # self.remove_last_fit()
        StyleWidget.is_opened = False
        StyleWidget._instance = None
        self.dock_widget.setVisible(False)

#
# if __name__ == "__main__":
#     import sys
#
#
#     def my_exception_hook(exctype, value, traceback):
#         # Print the error and traceback
#         print(exctype, value, traceback)
#         # Call the normal Exception hook after
#         sys._excepthook(exctype, value, traceback)
#         sys.exit(1)
#
#
#     from PyQt5.QtWidgets import QApplication
#
#     sys._excepthook = sys.excepthook
#
#     # Set the exception hook to our wrapping function
#     sys.excepthook = my_exception_hook
#
#     app = QtWidgets.QApplication(sys.argv)
#     Dialog = FitWidget(None, None)
#     # Dialog.show()
#     sys.exit(app.exec_())
