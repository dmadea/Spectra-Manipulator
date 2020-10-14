from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt, pyqtSignal
from dialogs.gui_fit_widget import Ui_Form

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox
import numpy as np

from spectrum import Spectrum

import pyqtgraph as pg

from logger import Logger

# from user_namespace import UserNamespace
from console import Console

# import lmfit
from lmfit import fit_report, report_fit, Minimizer, report_ci, conf_interval, conf_interval2d, Parameters
import matplotlib.pyplot as plt

import fitmodels
import userfitmodels
import inspect
import sys
from general_model import GeneralModel

from plotwidget import PlotWidget
from .fitresult import FitResult

import glob
import os

from settings import Settings


class FitWidget(QtWidgets.QWidget, Ui_Form):
    # static variables
    is_opened = False
    _instance = None

    # maximum number of parameters
    max_count = 30

    def __init__(self, dock_widget, accepted_func, spectrum=None, parent=None):
        super(FitWidget, self).__init__(parent)

        if FitWidget._instance is not None:
            PlotWidget.instance.removeItem(FitWidget._instance.lr)

        self.setupUi(self)

        self.dock_widget = dock_widget
        self.accepted_func = accepted_func

        self.fitted_params = None
        self.covariance_matrix = None
        self.errors = None
        self.fitted_spectrum = None
        self.residual_spectrum = None

        self.plot_fit = None
        self.plot_residuals = None

        # if spectrum is None - we have just open the widget for function plotting
        self.spectrum = spectrum

        self.plot_widget = PlotWidget.instance

        self.cbUpdateInitValues.setCheckState(Qt.Checked)

        f = 0.87
        x0, x1 = self.plot_widget.getViewBox().viewRange()[0]
        xy_dif = x1 - x0
        self.lr = pg.LinearRegionItem([x0 + (1 - f) * xy_dif, x0 + f * xy_dif],
                                      brush=QtGui.QBrush(QtGui.QColor(0, 255, 0, 20)),
                                      bounds=(self.spectrum.x_min(), self.spectrum.x_max())
                                      if self.spectrum is not None else None)
        self.lr.setZValue(-10)
        self.plot_widget.addItem(self.lr)

        # update region when the focus is lost and when the user presses enter
        self.leX0.returnPressed.connect(self.updateRegion)
        self.leX1.returnPressed.connect(self.updateRegion)
        self.leX0.focus_lost.connect(self.updateRegion)
        self.leX1.focus_lost.connect(self.updateRegion)

        self.lr.sigRegionChanged.connect(self.updatePlot)
        self.lr.sigRegionChangeFinished.connect(self.update_initial_values)
        self.updatePlot()

        self.sbParamsCount.valueChanged.connect(self.params_count_changed)
        self.btnFit.clicked.connect(self.fit)
        self.btnPlotFunction.clicked.connect(self.plot_function)
        self.btnPrintReport.clicked.connect(self.print_report)
        self.btnClearPlot.clicked.connect(self.clear_plot)
        # self.btnPrintDiffEquations.clicked.connect(self.print_diff_eq)
        self.btnOK.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)
        self.cbModel.currentIndexChanged.connect(self.model_changed)
        self.cbGenModels.currentIndexChanged.connect(self.cbGenModel_changed)
        self.btnBuildModel.clicked.connect(lambda: self.build_gen_model(True))
        self.cbShowBackwardRates.stateChanged.connect(self.cbShowBackwardRates_checked_changed)

        self.sbParamsCount.setMaximum(self.max_count)

        self.params_list = []
        self.lower_bound_list = []
        self.value_list = []
        self.upper_bound_list = []
        self.fixed_list = []
        self.error_list = []

        for i in range(self.max_count):
            self.params_list.append([QLineEdit(), QLineEdit()])
            self.lower_bound_list.append([QLineEdit(), QLineEdit()])
            self.value_list.append([QLineEdit(), QLineEdit()])
            self.upper_bound_list.append([QLineEdit(), QLineEdit()])
            self.fixed_list.append([QCheckBox(), QCheckBox()])
            self.error_list.append([QLineEdit(), QLineEdit()])
            self.params_list[i][1].setReadOnly(True)
            self.error_list[i][0].setReadOnly(True)
            self.error_list[i][1].setReadOnly(True)

            for j, layout in enumerate([self.predefGridLayout, self.cusModelGridLayout]):
                layout.addWidget(self.params_list[i][j], i + 1, 0, 1, 1)
                layout.addWidget(self.lower_bound_list[i][j], i + 1, 1, 1, 1)
                layout.addWidget(self.value_list[i][j], i + 1, 2, 1, 1)
                layout.addWidget(self.upper_bound_list[i][j], i + 1, 3, 1, 1)
                layout.addWidget(self.fixed_list[i][j], i + 1, 4, 1, 1)
                layout.addWidget(self.error_list[i][j], i + 1, 5, 1, 1)

            self.fixed_list[i][0].stateChanged.connect(self.fixed_checked_changed)
            self.fixed_list[i][1].stateChanged.connect(self.fixed_checked_changed)

            # if we are only plotting functions, disable some controls
            if self.spectrum is None:
                for j in (0, 1):
                    self.lower_bound_list[i][j].setEnabled(False)
                    self.upper_bound_list[i][j].setEnabled(False)
                    self.fixed_list[i][j].setEnabled(False)
                    self.error_list[i][j].setEnabled(False)

        if self.spectrum is None:
            self.btnFit.setEnabled(False)
            self.btnPrintReport.setEnabled(False)
            self.cbMethod.setEnabled(False)
            self.cbUpdateInitValues.setEnabled(False)

        # get all models from fitmodels, get classes that inherits from Model base class and sort them by name
        # and number of species
        classes = inspect.getmembers(sys.modules[fitmodels.__name__], inspect.isclass)
        tuples = filter(lambda tup: issubclass(tup[1], fitmodels._Model) and tup[1] is not fitmodels._Model, classes)
        # same load user defined fit models
        classes_usr = inspect.getmembers(sys.modules[userfitmodels.__name__], inspect.isclass)
        tuples_usr = filter(lambda tup: issubclass(tup[1], fitmodels._Model) and tup[1] is not fitmodels._Model,
                            classes_usr)
        # load models
        self.models = sorted(list(map(lambda tup: tup[1], tuples)) + list(map(lambda tup: tup[1], tuples_usr)),
                             key=lambda cls: cls.name)
        # fill the available models combo box with model names
        self.cbModel.addItems(map(lambda m: m.name, self.models))

        self.gen_models_paths = []
        for fpath in glob.glob(os.path.join(Settings.general_models_dir, '*.json'), recursive=True):
            fname = os.path.splitext(os.path.split(fpath)[1])[0]
            self.gen_models_paths.append(fpath)
            self.cbGenModels.addItem(fname)

        # abbr is name that is needed for lmfit.fitting method
        self.methods = [
            {'name': 'Trust Region Reflective method', 'abbr': 'least_squares'},
            {'name': 'Levenberg–Marquardt', 'abbr': 'leastsq'},
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
        self.current_general_model = None
        self.general_model_params = None
        self.fit_result = None
        self.plotted_functions = []
        self.plotted_function_spectra = []

        self.accepted = False
        FitWidget.is_opened = True
        FitWidget._instance = self

        self.dock_widget.parent().resizeDocks([self.dock_widget], [250], Qt.Vertical)
        self.dock_widget.titleBarWidget().setText(
            "Fit of {}".format(self.spectrum.name) if self.spectrum is not None else "Function plotter")
        self.dock_widget.setWidget(self)
        self.dock_widget.setVisible(True)

    # @staticmethod
    # def get_instance():
    #     return FitWidget._instance

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

    def cbGenModel_changed(self):
        self.current_general_model = GeneralModel.load(self.gen_models_paths[self.cbGenModels.currentIndex()])
        self.pteScheme.setPlainText(self.current_general_model.scheme)
        self.build_gen_model(load_scheme=False)

    def cbShowBackwardRates_checked_changed(self):
        if self.current_general_model is None:
            return

        self.build_gen_model(False)

    def build_gen_model(self, load_scheme=True):
        if load_scheme:
            self.current_general_model = GeneralModel.from_text(self.pteScheme.toPlainText())

        comps = self.current_general_model.get_compartments()
        init_comps_names = list(map(lambda n: f'c0_{n}', comps))
        rates = self.current_general_model.get_rates(self.cbShowBackwardRates.isChecked(),
                                                     append_values=True)

        n_params = 2 * len(comps) + len(rates)
        self.set_field_visible(n_params, 1)

        self.general_model_params = Parameters()

        for i, col in enumerate([init_comps_names, comps]):
            for j, name in enumerate(col):
                self.general_model_params.add(name, value=1 if j == 0 else 0,
                                              vary=bool(i),
                                              min=-np.inf,
                                              max=np.inf)

        for name, value in rates:
            self.general_model_params.add(name, value=value,
                                          vary=True,
                                          min=0,
                                          max=np.inf)

        for i, p in enumerate(self.general_model_params.values()):
            self.params_list[i][1].setText(p.name)
            self.value_list[i][1].setText(str(p.value))
            self.lower_bound_list[i][1].setText(str(p.min))
            self.upper_bound_list[i][1].setText(str(p.max))
            self.fixed_list[i][1].setChecked(not p.vary)

    def cbCustom_checked_changed(self):
        if self.cbCustom.isChecked():
            self.sbParamsCount.setEnabled(True)
            self.pteEquation.setReadOnly(False)
            for i in range(self.max_count):
                self.params_list[i][0].setReadOnly(False)
        else:
            self.sbParamsCount.setEnabled(False)
            self.pteEquation.setReadOnly(True)
            for i in range(self.max_count):
                self.params_list[i][0].setReadOnly(True)

    def update_initial_values(self):

        if not self.cbUpdateInitValues.isChecked():
            return

        if self.current_model is None:
            return

        if self.spectrum is None:
            # iterate parameters and fill the values
            for i, p in enumerate(self.current_model.params.values()):
                self.value_list[i][0].setText("{:.4g}".format(p.value))
            return

        x0, x1 = self.lr.getRegion()

        start_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x1) + 1

        x_data = self.spectrum.data[start_idx:end_idx, 0]
        y_data = self.spectrum.data[start_idx:end_idx, 1]

        self.current_model.initialize_values(x_data, y_data)

        # iterate parameters and fill the values
        for i, p in enumerate(self.current_model.params.values()):
            self.value_list[i][0].setText("{:.4g}".format(p.value))

    def model_changed(self):
        # initialize new model
        self.current_model = self.models[self.cbModel.currentIndex()]()
        params_count = self.current_model.par_count()
        self.sbParamsCount.setValue(params_count)
        self.pteEquation.setPlainText(self.current_model.get_func_string())

        # iterate parameters and fill the fields
        for i, p in enumerate(self.current_model.params.values()):
            self.params_list[i][0].setText(p.name)
            self.lower_bound_list[i][0].setText(str(p.min))
            self.upper_bound_list[i][0].setText(str(p.max))
            self.fixed_list[i][0].setChecked(not p.vary)

        self.update_initial_values()

    def clear_plot(self, keep_spectra=False):
        if self.plot_fit is not None and self.plot_residuals is not None:
            self.plot_widget.plotItem.removeItem(self.plot_fit)
            self.plot_widget.plotItem.removeItem(self.plot_residuals)
            try:
                self.plot_widget.legend.removeLastItem()
                self.plot_widget.legend.removeLastItem()
            except IndexError:
                pass

            self.plot_fit = None
            self.plot_residuals = None

        if len(self.plotted_functions) > 0:
            for item in self.plotted_functions:
                self.plot_widget.plotItem.removeItem(item)
            try:
                for i in range(len(self.plotted_functions)):
                    self.plot_widget.legend.removeLastItem()
            except IndexError:
                pass

            self.plotted_functions = []
            if not keep_spectra:
                self.plotted_function_spectra = []

    def fit(self):
        try:

            self._fit()
        except Exception as e:
            Logger.message(e.__str__())
            QMessageBox.warning(self, 'Fitting Error', e.__str__(), QMessageBox.Ok)

    # def remove_last_fit(self):
    #     if self.plot_fit is not None and self.plot_residuals is not None:
    #         self.plot_widget.plotItem.removeItem(self.plot_fit)
    #         self.plot_widget.plotItem.removeItem(self.plot_residuals)
    #         try:
    #             self.plot_widget.legend.removeLastItem()
    #             self.plot_widget.legend.removeLastItem()
    #         except IndexError:
    #             pass

    def _setup_model(self):
        import numpy as np
        from scipy.integrate import odeint
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
                                                                        self.pteEquation.toPlainText().replace('\n',
                                                                                                               '\n\t'))
            exec(func_command, locals(), globals())  # locals and globals had to be changed

            # change the blank function in abstract model by user defined
            self.current_model.func = user_defined_func

        if self.current_model is None:
            self.current_model = self.models[self.cbModel.currentIndex()]()

    def _fit(self):
        import numpy as np
        from scipy.integrate import odeint

        x0, x1 = self.lr.getRegion()

        start_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x0)
        end_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x1) + 1

        x_data = self.spectrum.data[start_idx:end_idx, 0]
        y_data = self.spectrum.data[start_idx:end_idx, 1]

        # setup model
        self._setup_model()

        # get params from field to current model
        for i in range(self.current_model.par_count()):
            param = self.params_list[i].text()
            self.current_model.params[param].value = float(self.value_list[i].text())
            self.current_model.params[param].min = float(self.lower_bound_list[i].text())
            self.current_model.params[param].max = float(self.upper_bound_list[i].text())
            self.current_model.params[param].vary = not self.fixed_list[i].isChecked()

        def residuals(params, x):
            y = self.current_model.wrapper_func(x, params)
            e = y - y_data
            if self.cbPropWeighting.isChecked():
                e /= y * y

            return e

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

        # self.remove_last_fit()
        self.clear_plot()

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

        self.fit_result = FitResult(result, minimizer, values_errors, self.current_model,
                                    data_item=self.spectrum, fit_item=self.fitted_spectrum,
                                    residuals_item=self.residual_spectrum)

    def plot_function(self):
        try:

            self._plot_function()
        except Exception as e:
            Logger.message(e.__str__())
            QMessageBox.warning(self, 'Plotting Error', e.__str__(), QMessageBox.Ok)

    def _plot_function(self):
        x0, x1 = self.lr.getRegion()

        # if this is a fit widget with real spectrum, use x range from that spectrum,
        # otherwise, use np.linspace with defined number of points in Settings
        if self.spectrum is not None:
            start_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x0)
            end_idx = Spectrum.find_nearest_idx(self.spectrum.data[:, 0], x1) + 1
            x_data = self.spectrum.data[start_idx:end_idx, 0]
        else:
            x_data = np.linspace(x0, x1, num=Settings.FP_num_of_points)

        self._setup_model()

        # get params from field to current model
        for i in range(self.current_model.par_count()):
            param = self.params_list[i].text()
            self.current_model.params[param].value = float(self.value_list[i].text())

        # calculate the y values according to our model
        y_data = self.current_model.wrapper_func(x_data, self.current_model.params)

        params = ""

        for i in range(self.current_model.par_count()):
            param = self.params_list[i].text()
            params += f"{param}={self.value_list[i].text()}, "

        params = params[:-2]  # remove last 2 characters, the space and comma
        name = f"Func plot: {params}"
        self.plotted_functions.append(self.plot_widget.plotItem.plot(x_data, y_data,
                                                                     pen=pg.mkPen(color=QColor(0, 0, 0, 255),
                                                                                  width=1),
                                                                     name=name))

        self.plotted_function_spectra.append(Spectrum.from_xy_values(x_data, y_data, name=name))

    def fixed_checked_changed(self, value):
        if self.spectrum is None:
            return

        checkbox = self.sender()  # get the checkbox that was clicked on

        i = 0
        for i, ch in enumerate(self.fixed_list):
            if ch == checkbox:
                break

        self.lower_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)
        self.upper_bound_list[i].setEnabled(True if value == Qt.Unchecked else False)

    def params_count_changed(self):

        count = int(self.sbParamsCount.value())

        self.set_field_visible(count, 0)

    def set_field_visible(self, n_params, idx=0):
        for i in range(self.max_count):
            visible = n_params > i

            self.params_list[i][idx].setVisible(visible)
            self.lower_bound_list[i][idx].setVisible(visible)
            self.value_list[i][idx].setVisible(visible)
            self.upper_bound_list[i][idx].setVisible(visible)
            self.fixed_list[i][idx].setVisible(visible)
            self.error_list[i][idx].setVisible(visible)


    def set_result(self):
        pass

    #
    # # TODO-->>>
    # def print_diff_eq(self):
    #     if self.current_model is None:
    #         return
    #
    #     # self.current_model._print_diff_equations()
    #
    #     # command = f"ax = plt.axes([0, 0, 0.1, 0.2])\nax.set_xticks([])\nax.set_yticks([])" \
    #     #     f"\nax.axis('off')\nplt.text(0.3, 0.4, \"${self.current_model._print_diff_equations()}$\", size=30)"
    #     #
    #     #
    #     # Console.execute_command(command)
    #     # #
    #     # Console.execute_command("plt\n%matplotlib inline")
    #     # self.current_model.print_diff_equations()

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
        # self.remove_last_fit()
        self.clear_plot(keep_spectra=True)
        self.print_report()

        self.accepted = True
        FitWidget.is_opened = False
        FitWidget._instance = None
        self.plot_widget.removeItem(self.lr)
        del self.lr
        self.dock_widget.setVisible(False)
        self.accepted_func()
        # super(FitWidget, self).accept()

    def reject(self):
        # self.remove_last_fit()
        self.clear_plot()
        FitWidget.is_opened = False
        FitWidget._instance = None
        self.plot_widget.removeItem(self.lr)
        del self.lr
        self.dock_widget.setVisible(False)

        # super(FitWidget, self).reject()


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
    Dialog = FitWidget(None, None)
    # Dialog.show()
    sys.exit(app.exec_())
