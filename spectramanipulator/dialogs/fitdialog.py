from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt
from .gui_fit import Ui_Dialog

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox
import numpy as np

import pyqtgraph as pg

from spectramanipulator import Logger, Spectrum

# from user_namespace import UserNamespace
from console import Console


# import lmfit
from lmfit import Minimizer
from .fitresult import FitResult

import fitmodels
import inspect


class FitDialog(QtWidgets.QDialog, Ui_Dialog):
    # static variables
    is_opened = False
    _instance = None

    # maximum number of parameters
    max_count = 20

    def __init__(self, spectrum, plot_widget, parent=None):
        super(FitDialog, self).__init__(parent, QtCore.Qt.WindowStaysOnTopHint)  # window stays on top
        self.setupUi(self)

        self.fitted_params = None
        self.covariance_matrix = None
        self.errors = None
        self.fitted_spectrum = None
        self.residual_spectrum = None

        self.plot_fit = None
        self.plot_residuals = None

        self.spectrum = spectrum
        self.plot_widget = plot_widget

        self.setWindowTitle("Fit of {}".format(self.spectrum.name))
        self.cbUpdateInitValues.setCheckState(Qt.Checked)

        f = 0.87
        x0, x1 = self.plot_widget.getViewBox().viewRange()[0]
        xy_dif = x1 - x0
        self.lr = pg.LinearRegionItem([x0 + (1 - f) * xy_dif, x0 + f * xy_dif],
                                      brush=QtGui.QBrush(QtGui.QColor(0, 255, 0, 20)),
                                      bounds=(self.spectrum.x_min(), self.spectrum.x_max()))
        self.lr.setZValue(-10)
        self.plot_widget.addItem(self.lr)

        self.leX0.returnPressed.connect(self.updateRegion)
        self.leX1.returnPressed.connect(self.updateRegion)

        self.lr.sigRegionChanged.connect(self.updatePlot)
        self.lr.sigRegionChangeFinished.connect(self.update_initial_values)
        self.updatePlot()

        self.sbParamsCount.valueChanged.connect(self.params_count_changed)
        self.btnFit.clicked.connect(self.fit)
        self.btnPrintReport.clicked.connect(self.print_report)
        self.btnPrintDiffEquations.clicked.connect(self.print_diff_eq)
        self.btnOK.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)
        self.cbModel.currentIndexChanged.connect(self.model_changed)

        self.sbParamsCount.setMaximum(self.max_count)

        self.params_list = []
        self.lower_bound_list = []
        self.value_list = []
        self.upper_bound_list = []
        self.fixed_list = []
        self.error_list = []

        for i in range(self.max_count):
            self.params_list.append(QLineEdit())
            self.lower_bound_list.append(QLineEdit())
            self.value_list.append(QLineEdit())
            self.upper_bound_list.append(QLineEdit())
            self.fixed_list.append(QCheckBox())
            self.error_list.append(QLineEdit())
            self.error_list[i].setReadOnly(True)

            self.gridLayout.addWidget(self.params_list[i], i + 1, 0, 1, 1)
            self.gridLayout.addWidget(self.lower_bound_list[i], i + 1, 1, 1, 1)
            self.gridLayout.addWidget(self.value_list[i], i + 1, 2, 1, 1)
            self.gridLayout.addWidget(self.upper_bound_list[i], i + 1, 3, 1, 1)
            self.gridLayout.addWidget(self.fixed_list[i], i + 1, 4, 1, 1)
            self.gridLayout.addWidget(self.error_list[i], i + 1, 5, 1, 1)

            self.fixed_list[i].stateChanged.connect(self.fixed_checked_changed)

        # get all models from fitmodels, get classes that inherits from Model base class and sort them by name
        # and number of species
        classes = inspect.getmembers(sys.modules[fitmodels.__name__], inspect.isclass)
        tuples = filter(lambda tup: issubclass(tup[1], fitmodels._Model) and tup[1] is not fitmodels._Model, classes)
        self.models = sorted(list(map(lambda tup: tup[1], tuples)), key=lambda cls: cls.name)
        # fill the available models combo box with model names
        self.cbModel.addItems(map(lambda m: m.name, self.models))

        self.methods = [
            {'name': 'Levenberg–Marquardt', 'abbr': 'leastsq'},
            {'name': 'Trust Region Reflective method', 'abbr': 'least_squares'},
            {'name': 'Differential evolution', 'abbr': 'differential_evolution'},
            {'name': 'Nelder-Mead, Simplex method (no error)', 'abbr': 'nelder'},
            {'name': 'L-BFGS-B (no error)', 'abbr': 'lbfgsb'},
            {'name': 'Powell (no error)', 'abbr': 'powell'}
        ]

        self.cbMethod.addItems(map(lambda m: m['name'], self.methods))

        self.cbCustom.stateChanged.connect(self.cbCustom_checked_changed)
        self.cbCustom_checked_changed()
        self.model_changed()

        self.current_model = None
        self.fit_result = None

        self.accepted = False
        FitDialog.is_opened = True
        FitDialog._instance = self

        self.show()
        self.exec()

    @staticmethod
    def get_instance():
        return FitDialog._instance

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else 0

    def updateRegion(self):
        try:
            x0, x1 = float(self.leX0.text()), float(self.leX1.text())
            self.lr.setRegion((x0, x1))
        except ValueError:
            pass

    def updatePlot(self):
        x0, x1 = self.lr.getRegion()
        self.leX0.setText("{:.4g}".format(x0))
        self.leX1.setText("{:.4g}".format(x1))

    def cbCustom_checked_changed(self):
        if self.cbCustom.isChecked():
            self.sbParamsCount.setEnabled(True)
            self.pteEquation.setReadOnly(False)
            for i in range(self.max_count):
                self.params_list[i].setReadOnly(False)
        else:
            self.sbParamsCount.setEnabled(False)
            self.pteEquation.setReadOnly(True)
            for i in range(self.max_count):
                self.params_list[i].setReadOnly(True)

    def update_initial_values(self):

        if not self.cbUpdateInitValues.isChecked():
            return

        if self.current_model is None:
            return

        x0, x1 = self.lr.getRegion()

        start_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x1) + 1

        x_data = self.spectrum.data[start_idx:end_idx, 0]
        y_data = self.spectrum.data[start_idx:end_idx, 1]

        self.current_model.initialize_values(x_data, y_data)

        # iterate parameters and fill the values
        for i, p in enumerate(self.current_model.params.values()):
            self.value_list[i].setText("{:.4g}".format(p.value))

    def model_changed(self):
        # initialize new model
        self.current_model = self.models[self.cbModel.currentIndex()]()
        params_count = self.current_model.par_count()
        self.sbParamsCount.setValue(params_count)
        self.pteEquation.setPlainText(self.current_model.get_func_string())

        # iterate parameters and fill the fields
        for i, p in enumerate(self.current_model.params.values()):
            self.params_list[i].setText(p.name)
            self.lower_bound_list[i].setText(str(p.min))
            self.upper_bound_list[i].setText(str(p.max))
            self.fixed_list[i].setChecked(not p.vary)

        self.update_initial_values()

    def fit(self):
        try:

            self._fit()
        except Exception as e:
            Logger.message(e.__str__())
            QMessageBox.warning(self, 'Fitting Error', e.__str__(), QMessageBox.Ok)

    def remove_last_fit(self):
        if self.plot_fit is not None and self.plot_residuals is not None:
            self.plot_widget.plotItem.removeItem(self.plot_fit)
            self.plot_widget.plotItem.removeItem(self.plot_residuals)
            try:
                self.plot_widget.legend.removeLastItem()
                self.plot_widget.legend.removeLastItem()
            except IndexError:
                pass

    def _fit(self):
        import numpy as np
        from scipy.integrate import odeint

        x0, x1 = self.lr.getRegion()

        start_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x1) + 1

        x_data = self.spectrum.data[start_idx:end_idx, 0]
        y_data = self.spectrum.data[start_idx:end_idx, 1]

        # custom model check box checked, we have to create own model from scratch
        if self.cbCustom.isChecked():

            self.current_model = fitmodels._Model()

            p_names = []

            for i in range(int(self.sbParamsCount.value())):
                param = self.params_list[i].text()
                p_names.append(param)
                self.current_model.params.add(param)

            # create new function, fill params names after x and body of the function (take care of proper indentation)
            func_command = "def user_defined_func(x, {}):\n\t{}".format(','.join(p_names),
                self.pteEquation.toPlainText().replace('\n', '\n\t'))
            exec(func_command, locals(), globals())  # locals and globals had to be changed

            # change the blank function in abstract model by user defined
            self.current_model.func = user_defined_func

        if self.current_model is None:
            self.current_model = self.models[self.cbModel.currentIndex()]()

        # get params from field to current model
        for i in range(self.current_model.par_count()):
            param = self.params_list[i].text()
            self.current_model.params[param].value = float(self.value_list[i].text())
            self.current_model.params[param].min = float(self.lower_bound_list[i].text())
            self.current_model.params[param].max = float(self.upper_bound_list[i].text())
            self.current_model.params[param].vary = not self.fixed_list[i].isChecked()

        def residuals(params, x):
            return self.current_model.wrapper_func(x, params) - y_data

        method = self.methods[self.cbMethod.currentIndex()]['abbr']

        minimizer = Minimizer(residuals, self.current_model.params, fcn_args=(x_data,))
        result = minimizer.minimize(method=method)  # fit

        self.current_model.params = result.params

        # fill fields
        for i in range(self.current_model.par_count()):
            param = self.params_list[i].text()

            self.value_list[i].setText("{:.4g}".format(self.current_model.params[param].value))
            error = self.current_model.params[param].stderr
            self.error_list[i].setText("{:.4g}".format(error) if error is not None else '')

        y_fit_data = self.current_model.wrapper_func(x_data, self.current_model.params)
        y_residuals = y_data - y_fit_data

        self.remove_last_fit()

        self.plot_fit = self.plot_widget.plotItem.plot(x_data, y_fit_data,
                                                       pen=pg.mkPen(color=QColor(0, 0, 0, 200), width=2.5),
                                                       name="Fit of {}".format(self.spectrum.name))

        self.plot_residuals = self.plot_widget.plotItem.plot(x_data, y_residuals,
                                                             pen=pg.mkPen(color=QColor(255, 0, 0, 150), width=1),
                                                             name="Residuals of {}".format(self.spectrum.name))

        self.fitted_spectrum = Spectrum.from_xy_values(x_data, y_fit_data,
                                                       name="Fit of {}".format(self.spectrum.name),
                                                       color='black', line_width=2.5, line_type=Qt.SolidLine)

        self.residual_spectrum = Spectrum.from_xy_values(x_data, y_residuals,
                                                         name="Residuals of {}".format(self.spectrum.name),
                                                         color='red', line_width=1, line_type=Qt.SolidLine)

        values_errors = np.zeros((self.current_model.par_count(), 2), dtype=np.float64)

        for i in range(self.current_model.par_count()):
            param = self.params_list[i].text()
            values_errors[i, 0] = self.current_model.params[param].value
            error = self.current_model.params[param].stderr
            values_errors[i, 1] = error if error is not None else 0

        self.fit_result = FitResult(result, minimizer, values_errors,
                                    data_item=self.spectrum, fit_item=self.fitted_spectrum,
                                    residuals_item=self.residual_spectrum)

    def fixed_checked_changed(self, value):
        checkbox = self.sender()

        i = 0
        for i, ch in enumerate(self.fixed_list):
            if ch == checkbox:
                break

        self.lower_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)
        self.upper_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)

    def params_count_changed(self):

        count = int(self.sbParamsCount.value())

        for i in range(self.max_count):
            visible = count > i

            self.params_list[i].setVisible(visible)
            self.lower_bound_list[i].setVisible(visible)
            self.value_list[i].setVisible(visible)
            self.upper_bound_list[i].setVisible(visible)
            self.fixed_list[i].setVisible(visible)
            self.error_list[i].setVisible(visible)

    def set_result(self):
        pass

    # TODO-->>>
    def print_diff_eq(self):
        if self.current_model is None:
            return

        # self.current_model._print_diff_equations()

        # command = f"ax = plt.axes([0, 0, 0.1, 0.2])\nax.set_xticks([])\nax.set_yticks([])" \
        #     f"\nax.axis('off')\nplt.text(0.3, 0.4, \"${self.current_model._print_diff_equations()}$\", size=30)"
        #
        #
        # Console.execute_command(command)
        # #
        # Console.execute_command("plt\n%matplotlib inline")
        # self.current_model.print_diff_equations()

    def print_report(self):
        if self.fit_result is None:
            return
        try:
            if self.current_model is not None:
                report = "Fitting of '{}' with model '{}' succeeded. " \
                         "You can find fit results in variable " \
                         "'fit'.\n{}\n".format(self.spectrum.name, self.current_model.name,
                                               self.fit_result.report(print=False))

                Logger.console_message(report)
                Console.push_variables({'fit': self.fit_result})

        except Exception as e:
            Logger.message(e.__str__())
            QMessageBox.warning(self, 'Printing Error', e.__str__(), QMessageBox.Ok)

    def accept(self):
        self.remove_last_fit()
        self.print_report()

        self.accepted = True
        FitDialog.is_opened = False
        FitDialog._instance = None
        self.plot_widget.removeItem(self.lr)
        del self.lr
        super(FitDialog, self).accept()

    def reject(self):
        self.remove_last_fit()
        FitDialog.is_opened = False
        FitDialog._instance = None
        self.plot_widget.removeItem(self.lr)
        del self.lr
        super(FitDialog, self).reject()


if __name__ == "__main__":
    import sys


    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    from PyQt5.QtWidgets import QApplication

    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    app = QtWidgets.QApplication(sys.argv)
    Dialog = FitDialog(None, None)
    # Dialog.show()
    sys.exit(app.exec_())
