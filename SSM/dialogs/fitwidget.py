from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt, pyqtSignal
from .gui_fit_widget import Ui_Form

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox, QFileDialog
import numpy as np

from SSM import Spectrum, Logger, Console, Settings

import pyqtgraph as pg

# from user_namespace import UserNamespace

from lmfit import fit_report, report_fit, Minimizer, report_ci, conf_interval, conf_interval2d, Parameters
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import SSM.fitmodels
import SSM.userfitmodels
import inspect
import sys
from ..general_model import GeneralModel

from ..plotwidget import PlotWidget
from .fitresult import FitResult

from ..utils.syntax_highlighter import PythonHighlighter, KineticModelHighlighter

import glob
import os

from ..misc import int_default_color_scheme


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
        self.btnSaveCustomModel.clicked.connect(self.save_general_model_as)

        self.sbParamsCount.setMaximum(self.max_count)

        self.pteEquation_highlighter = PythonHighlighter(self.pteEquation.document())
        self.ppteScheme_highlighter = KineticModelHighlighter(self.pteScheme.document())

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
        self.update_general_models()

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

    def update_general_models(self):
        # curr_idx = self.cbGenModels.currentIndex() if len(self.cbGenModels) > 0 else None
        self.cbGenModels.clear()
        self.gen_models_paths = []
        for fpath in glob.glob(os.path.join(Settings.general_models_dir, '*.json'), recursive=True):
            fname = os.path.splitext(os.path.split(fpath)[1])[0]
            self.gen_models_paths.append(fpath)
            self.cbGenModels.addItem(fname)
        #
        # if curr_idx is not None:
        #     self.cbGenModels.setCurrentIndex(curr_idx)

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else 0

    def updateRegion(self):
        try:
            x0, x1 = float(self.leX0.text()), float(self.leX1.text())
            self.lr.setRegion((x0, x1))
        except ValueError:
            pass

    def save_general_model_as(self):
        if self.current_general_model is None:
            return
        curr_model_path = self.gen_models_paths[self.cbGenModels.currentIndex()]
        fil = "Json (*.json)"

        filepath = QFileDialog.getSaveFileName(caption="Save Custom Kinetic Model",
                                               directory=curr_model_path,
                                               filter=fil, initialFilter=fil)
        if filepath[0] == '':
            return

        try:
            init, _, rates, _ = self.get_params_from_fields()
            self.current_general_model.scheme = self.pteScheme.toPlainText()
            self.current_general_model.set_rates(rates)
            self.current_general_model.initial_conditions = dict(
                zip(self.current_general_model.get_compartments(), init))
            self.current_general_model.save(filepath[0])
        except Exception as e:
            Logger.message(e.__str__())
            QMessageBox.critical(self, 'Saving error', 'Error, first build the model.', QMessageBox.Ok)
            return

        self.update_general_models()
        k = 0
        for k, fpath in enumerate(self.gen_models_paths):
            fname = os.path.splitext(os.path.split(fpath)[1])[0]
            if fname == os.path.splitext(os.path.split(filepath[0])[1])[0]:
                break

        # set saved model as current index
        self.cbGenModels.setCurrentIndex(k)

    def get_params_from_fields(self):
        if self.current_general_model is None:
            return

        n_comps = len(self.current_general_model.get_compartments())
        n_params = len(self.current_general_model.elem_reactions)

        init_cond = np.empty(n_comps, dtype=np.float64)
        coefs = np.empty(n_comps, dtype=np.float64)
        rates = np.zeros((n_params, 2), dtype=np.float64)

        for i, (i_idx, coef_idx) in enumerate(zip(range(n_comps), range(n_comps, 2*n_comps))):
            init_cond[i] = float(self.value_list[i_idx][1].text())
            coefs[i] = float(self.value_list[coef_idx][1].text())

        show_bw_rates = self.cbShowBackwardRates.isChecked()
        last_idx = 2*n_comps + n_params * (2 if show_bw_rates else 1)
        forward_idxs = range(2*n_comps, last_idx, (2 if show_bw_rates else 1))

        for i, idx in enumerate(forward_idxs):
            rates[i, 0] = float(self.value_list[idx][1].text())
            if show_bw_rates:
                rates[i, 1] = float(self.value_list[idx + 1][1].text())

        return init_cond, coefs, rates, float(self.value_list[last_idx][1].text())

    def get_values_from_params(self, params):

        n_comps = len(self.current_general_model.get_compartments())
        n_params = len(self.current_general_model.elem_reactions)

        init_cond = np.empty(n_comps, dtype=np.float64)
        coefs = np.empty(n_comps, dtype=np.float64)
        rates = np.zeros((n_params, 2), dtype=np.float64)

        l_params = list(params.values())

        for i, (i_idx, coef_idx) in enumerate(zip(range(n_comps), range(n_comps, 2*n_comps))):
            init_cond[i] = l_params[i_idx].value
            coefs[i] = l_params[coef_idx].value

        show_bw_rates = self.cbShowBackwardRates.isChecked()
        forward_idxs = range(2*n_comps, 2*n_comps + n_params * (2 if show_bw_rates else 1), (2 if show_bw_rates else 1))

        for i, idx in enumerate(forward_idxs):
            rates[i, 0] = l_params[idx].value
            if show_bw_rates:
                rates[i, 1] = l_params[idx + 1].value

        return init_cond, coefs, rates, l_params[-1]  # last is intercept

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

        self.build_gen_model(load_scheme=False)

    def build_gen_model(self, load_scheme=True):
        if load_scheme:
            try:
                self.current_general_model = GeneralModel.from_text(self.pteScheme.toPlainText())
            except Exception as e:
                Logger.message(e.__str__())
                QMessageBox.critical(self, 'Build failed', f'Invalid model:\n\n{e.__str__()}', QMessageBox.Ok)
                return

        self.current_general_model.build_func()

        comps = self.current_general_model.get_compartments()
        init_comps_names = list(map(lambda n: f'[{n}]0', comps))
        rates = self.current_general_model.get_rates(self.cbShowBackwardRates.isChecked(),
                                                     append_values=True)

        n_params = 2 * len(comps) + len(rates) + 1  # + intercept
        self.set_field_visible(n_params, 1)

        self.general_model_params = Parameters()

        k = 0
        # initial conditions
        for init in self.current_general_model.initial_conditions.values():
            self.general_model_params.add(f'param{k}', value=init,
                                          vary=False,
                                          min=0,
                                          max=np.inf)
            k += 1
        # coefficients
        for i in range(len(comps)):
            self.general_model_params.add(f'param{k}', value=1 if i == 0 else 0,
                                          vary=True if i == 0 else False,
                                          min=-np.inf,
                                          max=np.inf)
            k += 1

        # rates
        rates_names = []
        for name, value in rates:
            rates_names.append(name)
            self.general_model_params.add(f'param{k}', value=value,
                                          vary=True,
                                          min=0,
                                          max=np.inf)
            k += 1

        # add intercept
        self.general_model_params.add('y0', value=0, vary=True, min=-np.inf, max=np.inf)

        for i, (p, name) in enumerate(zip(self.general_model_params.values(),
                                          init_comps_names + comps + rates_names + ['y0'])):
            self.params_list[i][1].setText(name)
            self.value_list[i][1].setText(f"{p.value:.5g}")
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

        tab_idx = self.tabWidget.currentIndex()

        if tab_idx == 0:
            self._setup_model()

        # fill the parameters from fields
        for i, p in enumerate((self.current_model.params if tab_idx == 0 else self.general_model_params).values()):
            p.value = float(self.value_list[i][tab_idx].text())
            p.min = float(self.lower_bound_list[i][tab_idx].text())
            p.max = float(self.upper_bound_list[i][tab_idx].text())
            p.vary = not self.fixed_list[i][tab_idx].isChecked()

        def y_fit(params):
            if tab_idx == 0:
                y = self.current_model.wrapper_func(x_data, params)
            else:
                init, coefs, rates, y0 = self.get_values_from_params(params)
                sol = self._simul_custom_model(init, rates, x_data)
                y = (coefs * sol).sum(axis=1, keepdims=False) + y0
            return y

        def residuals(params):
            y = y_fit(params)
            e = y - y_data
            if self.cbPropWeighting.isChecked():
                e /= y * y

            return e

        minimizer = Minimizer(residuals, self.current_model.params if tab_idx == 0 else self.general_model_params)

        method = self.methods[self.cbMethod.currentIndex()]['abbr']
        result = minimizer.minimize(method=method)  # fit

        if tab_idx == 0:
            self.current_model.params = result.params
        else:
            self.general_model_params = result.params

        # fill fields
        values_errors = np.zeros((len(result.params), 2), dtype=np.float64)
        for i, p in enumerate(result.params.values()):
            values_errors[i, 0] = p.value
            values_errors[i, 1] = p.stderr if p.stderr is not None else 0

            self.value_list[i][tab_idx].setText(f"{p.value:.4g}")
            self.error_list[i][tab_idx].setText(f"{p.stderr:.4g}" if p.stderr else '')

        y_fit_data = y_fit(result.params)
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

        self.fit_result = FitResult(result, minimizer, values_errors, (self.current_model if tab_idx == 0 else self.current_general_model),
                                    data_item=self.spectrum, fit_item=self.fitted_spectrum,
                                    residuals_item=self.residual_spectrum)

    def plot_function(self):

        # self._plot_function()
        try:

            self._plot_function()
        except Exception as e:
            Logger.message(e.__str__())
            QMessageBox.warning(self, 'Plotting Error', e.__str__(), QMessageBox.Ok)

    def _simul_custom_model(self, j, rates, x_data):
        if x_data[0] > 0:  # initial conditions are valid for time=0
            n = 100  # prepend x values with 100 points if not starting with zero time
            x_prepended = np.concatenate((np.linspace(0, x_data[0], n, endpoint=False), x_data))
            return odeint(self.current_general_model.func, j, x_prepended, args=(rates,))[n:, :]

        elif x_data[0] < 0:
            x_pos = x_data[x_data >= 0]  # find x >= 0
            sol = np.zeros((x_data.shape[0], j.shape[0]), dtype=np.float64)
            if x_pos.shape[0] > 1:  # simulate only for at least 2 positive values
                sol[(x_data < 0).sum():, :] = self._simul_custom_model(j, rates, x_pos)  # use recursion here

            return sol

        # for x_data[0] == 0
        return odeint(self.current_general_model.func, j, x_data, args=(rates,))

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

        tab_idx = self.tabWidget.currentIndex()

        if tab_idx == 0:  # equation-based model
            self._setup_model()

            par_count = self.current_model.par_count()

            # get params from field to current model
            for i in range(par_count):
                param = self.params_list[i][0].text()
                self.current_model.params[param].value = float(self.value_list[i][0].text())

            # calculate the y values according to our model
            y_data = self.current_model.wrapper_func(x_data, self.current_model.params)
        else:  # custom kinetic model
            if self.current_general_model is None:
                self.cbGenModel_changed()

            init, coefs, rates, y0 = self.get_params_from_fields()
            par_count = 2 * init.shape[0] + rates.shape[0]
            n_params = len(init)

            sol = self._simul_custom_model(init, rates, x_data)

            if self.cbPlotAllComps.isChecked():
                spectra = []
                for i in range(n_params):
                    name = self.params_list[i+n_params][tab_idx].text()
                    self.plotted_functions.append(self.plot_widget.plotItem.plot(x_data, sol[:, i],
                                                                             pen=pg.mkPen(color=int_default_color_scheme(i),
                                                                                          width=1),
                                                                             name=name))
                    spectra.append(Spectrum.from_xy_values(x_data, sol[:, i], name=name))
                self.plotted_function_spectra.append(spectra)
                return

            y_data = (sol * coefs).sum(axis=1, keepdims=False) + y0

        params = ', '.join([f"{self.params_list[i][tab_idx].text()}={self.value_list[i][tab_idx].text()}" for i in range(par_count)])

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

        idx = self.tabWidget.currentIndex()

        i = 0
        for i, ch in enumerate(self.fixed_list[idx]):
            if ch == checkbox:
                break

        enabled = True if value == Qt.Unchecked else False

        self.lower_bound_list[i][idx].setEnabled(enabled)
        self.upper_bound_list[i][idx].setEnabled(enabled)

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

    def print_report(self):
        if self.fit_result is None:
            return
        try:
            report = "Fitting of '{}' succeeded. " \
                     "You can find fit results in variable " \
                     "'fit'.\n{}\n".format(self.spectrum.name,
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
