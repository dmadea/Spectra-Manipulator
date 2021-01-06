from PyQt5 import QtWidgets

from PyQt5.QtCore import Qt
from .gui_fit_widget import Ui_Form

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox, QFileDialog
import numpy as np

from spectramanipulator.settings import Settings
from spectramanipulator.logger import Logger
from spectramanipulator.treeview.item import SpectrumItem, SpectrumItemGroup
from spectramanipulator.console import Console
from spectramanipulator.spectrum import fi
import spectramanipulator

import pyqtgraph as pg

# from user_namespace import UserNamespace

from lmfit import Minimizer, Parameters
from scipy.integrate import odeint

import spectramanipulator.fitting.fitmodels as fitmodels

import inspect
from ..general_model import GeneralModel

from ..plotwidget import PlotWidget
from .fitresult import FitResult

from ..utils.syntax_highlighter import PythonHighlighter, KineticModelHighlighter

import glob
import os
import time

import sys

from ..misc import int_default_color_scheme
from .combobox_cb import ComboBoxCB
from copy import deepcopy

import math

def is_nan_or_inf(value):
    return math.isnan(value) or math.isinf(value)



class FitWidget(QtWidgets.QWidget, Ui_Form):
    # static variables
    is_opened = False
    _instance = None

    # maximum number of parameters
    max_param_count = 25

    def __init__(self, dock_widget, accepted_func, node: [SpectrumItem, SpectrumItemGroup] = None, parent=None):
        super(FitWidget, self).__init__(parent)

        # if FitWidget._instance is not None:
        #     PlotWidget._instance.removeItem(FitWidget._instance.lr)

        self.setupUi(self)

        self.dock_widget = dock_widget
        self.accepted_func = accepted_func

        self.fitted_params = None
        self.covariance_matrix = None
        self.errors = None

        # if spectrum is None - we have just open the widget for function plotting
        self.node = None
        if node is not None:
            # could be spectrum or a SpectrumItemGroup for multi and batch fitting
            self.node = [node] if len(node.children) == 0 else node.children

        self.lr = None
        self.show_region_checked_changed()

        # update region when the focus is lost and when the user presses enter
        self.leX0.returnPressed.connect(self.update_region)
        self.leX1.returnPressed.connect(self.update_region)
        self.leX0.focus_lost.connect(self.update_region)
        self.leX1.focus_lost.connect(self.update_region)

        # self.lr.sigRegionChangeFinished.connect(self.update_initial_values)
        self.update_region_text_values()

        self.btnFit.clicked.connect(self.fit)
        self.btnSimulateModel.clicked.connect(self.simulate_model)
        self.btnPrintReport.clicked.connect(self.print_report)
        self.btnClearPlot.clicked.connect(self.clear_plot)
        # self.btnPrintDiffEquations.clicked.connect(self.print_diff_eq)
        self.btnOK.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)
        self.cbGenModels.currentIndexChanged.connect(self.cbGenModel_changed)
        self.btnBuildModel.clicked.connect(lambda: self.build_gen_model(True))
        self.cbShowBackwardRates.stateChanged.connect(self.cbShowBackwardRates_checked_changed)
        self.btnSaveCustomModel.clicked.connect(self.save_general_model_as)
        self.sbSpeciesCount.valueChanged.connect(lambda: self.n_spec_changed())
        self.cbShowResiduals.toggled.connect(self._replot)
        self.cbFitBlackColor.toggled.connect(self._replot)
        self.cbShowRegion.toggled.connect(self.show_region_checked_changed)

        # specific kinetic model setup

        self.ppteScheme_highlighter = KineticModelHighlighter(self.pteScheme.document())

        self.model_options_grid_layout = QtWidgets.QGridLayout(self)
        self.verticalLayout.addLayout(self.model_options_grid_layout)
        self.pred_model_indep_params_layout = self.create_params_layout()
        self.verticalLayout.addLayout(self.pred_model_indep_params_layout)

        self.cbPredefModelDepParams = ComboBoxCB(self)
        self.cbPredefModelDepParams_changing = False
        self.cbPredefModelDepParams.check_changed.connect(self.cbPredefModelDepParams_check_changed)
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.addWidget(QtWidgets.QLabel('Selected experiment-dependent parameters:'))
        hbox.addWidget(self.cbPredefModelDepParams)
        self.verticalLayout.addLayout(hbox)

        self.tab_widget_pred_model = QtWidgets.QTabWidget(self)
        self.verticalLayout.addWidget(self.tab_widget_pred_model)

        self.pred_species_hlayouts = []
        self.pred_model_dep_param_layouts = []

        if self.node is not None:
            for spectrum in self.node:
                widget, species_hlayout, params_layout = self.create_tab_widget(self.tab_widget_pred_model)
                self.pred_species_hlayouts.append(species_hlayout)
                self.pred_model_dep_param_layouts.append(params_layout)

                self.tab_widget_pred_model.addTab(widget, spectrum.name)

        vals_temp = dict(params=[], lower_bounds=[], values=[], upper_bounds=[], fixed=[], errors=[])
        self.pred_param_fields = dict(exp_independent=deepcopy(vals_temp),
                                      exp_dependent=[deepcopy(vals_temp) for _ in range(len(self.node))])

        self.pred_spec_visible_fields = [[] for _ in range(len(self.node))]

        def add_widgets(dict_fields: dict, param_layout: QtWidgets.QGridLayout):
            for i in range(self.max_param_count):
                par = QLineEdit()  # parameter text field
                lb = QLineEdit()  # lower bound text field
                val = QLineEdit()  # value text field
                ub = QLineEdit()  # upper bound text field
                fix = QCheckBox()  # fixed checkbox
                err = QLineEdit()  # std err text field

                par.setReadOnly(True)
                err.setReadOnly(True)

                # set par as an attribute to all other field to be able to access the param field from them
                lb.par = par
                val.par = par
                ub.par = par
                fix.par = par

                # set lower and upper bound fields as tag for fix field to be able to handle fix field changes
                fix.lb = lb
                fix.ub = ub

                par.setVisible(False)
                lb.setVisible(False)
                val.setVisible(False)
                ub.setVisible(False)
                fix.setVisible(False)
                err.setVisible(False)

                # update model param upon field change
                lb.textChanged.connect(lambda value: self.transfer_param_to_model('min', value))
                val.textChanged.connect(lambda value: self.transfer_param_to_model('value', value))
                ub.textChanged.connect(lambda value: self.transfer_param_to_model('max', value))
                fix.toggled.connect(lambda value: self.transfer_param_to_model('vary', not value))

                # put references to dictionary
                dict_fields['params'].append(par)
                dict_fields['lower_bounds'].append(lb)
                dict_fields['values'].append(val)
                dict_fields['upper_bounds'].append(ub)
                dict_fields['fixed'].append(fix)
                dict_fields['errors'].append(err)

                # add widgets to layout
                param_layout.addWidget(par, i + 1, 0, 1, 1)
                param_layout.addWidget(lb, i + 1, 1, 1, 1)
                param_layout.addWidget(val, i + 1, 2, 1, 1)
                param_layout.addWidget(ub, i + 1, 3, 1, 1)
                param_layout.addWidget(fix, i + 1, 4, 1, 1)
                param_layout.addWidget(err, i + 1, 5, 1, 1)

        def set_tab_order(dict_fields: dict):
            for i in range(self.max_param_count):
                for key in dict_fields.keys():
                    for first, second in zip(dict_fields[key][:-1], dict_fields[key][1:]):
                        self.setTabOrder(first, second)

        add_widgets(self.pred_param_fields['exp_independent'], self.pred_model_indep_params_layout)
        set_tab_order(self.pred_param_fields['exp_independent'])

        for i in range(len(self.node)):
            add_widgets(self.pred_param_fields['exp_dependent'][i], self.pred_model_dep_param_layouts[i])
            set_tab_order(self.pred_param_fields['exp_dependent'][i])

            # setup species visible checkboxes
            for _ in range(self.max_param_count):
                cb = QCheckBox('')
                cb.exp_index = i
                cb.setVisible(False)
                cb.toggled.connect(self.species_visible_checkbox_toggled)
                self.pred_spec_visible_fields[i].append(cb)
                self.pred_species_hlayouts[i].addWidget(cb)
            self.pred_species_hlayouts[i].addSpacerItem(QtWidgets.QSpacerItem(1, 1,
                                                        QtWidgets.QSizePolicy.Expanding,
                                                        QtWidgets.QSizePolicy.Fixed))

        self.setting_params = False

        # get all models from fitmodels, get classes that inherits from Model base class and sort them by name
        # and number of species
        classes = inspect.getmembers(sys.modules[fitmodels.__name__], inspect.isclass)
        tuples = filter(lambda tup: issubclass(tup[1], fitmodels.Model) and tup[1] is not fitmodels.Model, classes)
        # same load user defined fit models
        # classes_usr = inspect.getmembers(sys.modules[userfitmodels.__name__], inspect.isclass)
        # tuples_usr = filter(lambda tup: issubclass(tup[1], fitmodels.Model) and tup[1] is not fitmodels.Model,
        #                     classes_usr)
        # load models
        self.models = sorted(list(map(lambda tup: tup[1], tuples)), key=lambda cls: cls.name)
        # fill the available models combo box with model names
        self.cbModel.addItems(map(lambda m: m.name, self.models))

        self.cbModel.currentIndexChanged.connect(self.model_changed)

        self.gen_models_paths = []
        self.update_general_models()

        self.res_weights = [
            {'description': 'No weighting', 'func': lambda res, y: res},
            {'description': 'Proportional weighting: residual *= 1/y^2', 'func': lambda res, y: res / (y * y)},
        ]

        self.cbResWeighting.addItems(map(lambda m: m['description'], self.res_weights))

        # abbr is name that is needed for lmfit.fitting method
        self.methods = [
            {'name': 'Trust Region Reflective method', 'abbr': 'least_squares'},
            {'name': 'Levenberg–Marquardt', 'abbr': 'leastsq'},
            {'name': 'Nelder-Mead, Simplex method (no error)', 'abbr': 'nelder'},
            {'name': 'Differential evolution', 'abbr': 'differential_evolution'},
            {'name': 'L-BFGS-B (no error)', 'abbr': 'lbfgsb'},
            {'name': 'Powell (no error)', 'abbr': 'powell'}
        ]

        self.cbMethod.addItems(map(lambda m: m['name'], self.methods))

        self.current_model = None
        self.current_general_model = None
        self.general_model_params = None
        self.fit_result = None
        self.fits = []
        self.residuals = []

        self.accepted = False
        FitWidget.is_opened = True
        FitWidget._instance = self

        self.dock_widget.parent().resizeDocks([self.dock_widget], [250], Qt.Vertical)
        self.title_text = 'Function plotter'
        if self.node is not None:
            self.title_text = 'Fit of {}'.format(self.node[0].parent.name if len(self.node) > 0 else self.node[0].name)
        self.dock_widget.titleBarWidget().setText(self.title_text)
        self.dock_widget.setWidget(self)
        self.dock_widget.setVisible(True)

        self.model_changed()

    def show_region_checked_changed(self):
        if not self.cbShowRegion.isChecked():
            PlotWidget.remove_linear_region()
            self.lr = None
            return

        bounds = None
        if self.node is not None:
            # find the max range for all of the data

            x0 = self.node[0].data[0, 0]
            x1 = self.node[0].data[-1, 0]

            for item in self.node[1:]:
                new_x0 = item.data[0, 0]
                new_x1 = item.data[-1, 0]
                if new_x0 < x0:
                    x0 = new_x0
                if new_x1 > x1:
                    x1 = new_x1
            x0 = -1 if is_nan_or_inf(x0) else x0
            x1 = 1 if is_nan_or_inf(x1) else x1

            bounds = (x0, x1)

        self.lr = PlotWidget.add_linear_region(bounds=bounds, z_value=1e8)
        self.update_region()
        self.lr.sigRegionChanged.connect(self.update_region_text_values)

    def update_region_text_values(self):
        x0, x1 = self.lr.getRegion()
        self.leX0.setText("{:.4g}".format(x0))
        self.leX1.setText("{:.4g}".format(x1))

    def update_region(self):
        if self.lr is None:
            return

        try:
            x0, x1 = float(self.leX0.text()), float(self.leX1.text())
            self.lr.setRegion((x0, x1))
        except ValueError:
            pass

    def create_tab_widget(self, tab_widget):
        widget = QtWidgets.QWidget(tab_widget)
        vl = QtWidgets.QVBoxLayout(widget)
        vl.addWidget(QtWidgets.QLabel('Species visible:'))
        species_hlayout = QtWidgets.QHBoxLayout(widget)
        vl.addLayout(species_hlayout)
        # vl.addWidget(QtWidgets.QLabel('Experiment-dependent parameters:'))
        params_layout = self.create_params_layout(widget)
        vl.addLayout(params_layout)

        widget.setLayout(vl)

        return widget, species_hlayout, params_layout

    def create_params_layout(self, parent=None):
        params_layout_template = QtWidgets.QGridLayout(self if parent is None else parent)
        param_label = QtWidgets.QLabel('Param')
        lb_label = QtWidgets.QLabel('Lower\nBound')
        val_label = QtWidgets.QLabel('\u2264 Value \u2264')  # <= Value <=
        ub_label = QtWidgets.QLabel('Upper\nBound')
        fix_label = QtWidgets.QLabel('Fixed')
        err_label = QtWidgets.QLabel('Error')
        param_label.setAlignment(Qt.AlignCenter)
        lb_label.setAlignment(Qt.AlignCenter)
        val_label.setAlignment(Qt.AlignCenter)
        ub_label.setAlignment(Qt.AlignCenter)
        fix_label.setAlignment(Qt.AlignCenter)
        err_label.setAlignment(Qt.AlignCenter)
        params_layout_template.addWidget(param_label, 0, 0, 1, 1)
        params_layout_template.addWidget(lb_label, 0, 1, 1, 1)
        params_layout_template.addWidget(val_label, 0, 2, 1, 1)
        params_layout_template.addWidget(ub_label, 0, 3, 1, 1)
        params_layout_template.addWidget(fix_label, 0, 4, 1, 1)
        params_layout_template.addWidget(err_label, 0, 5, 1, 1)

        return params_layout_template

    def update_general_models(self):
        # curr_idx = self.cbGenModels.currentIndex() if len(self.cbGenModels) > 0 else None
        self.cbGenModels.clear()
        self.gen_models_paths = []

        module_path = os.path.abspath(spectramanipulator.__file__)  # get the absolute path of a module
        fpath = os.path.join(os.path.dirname(module_path), Settings.general_models_dir, '*.json')

        for fpath in glob.glob(fpath, recursive=True):
            fname = os.path.splitext(os.path.split(fpath)[1])[0]
            self.gen_models_paths.append(fpath)
            self.cbGenModels.addItem(fname)

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else 0

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


    def cbGenModel_changed(self):
        pass
        # self.current_general_model = GeneralModel.load(self.gen_models_paths[self.cbGenModels.currentIndex()])
        # self.pteScheme.setPlainText(self.current_general_model.scheme)
        # self.build_gen_model(load_scheme=False)

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

        self.set_lower_upper_enabled()

    def cbCustom_checked_changed(self):
        if self.cbCustom.isChecked():
            self.sbParamsCount.setEnabled(True)
            self.pteEquation.setReadOnly(False)
            for i in range(self.max_param_count):
                self.params_list[i][0].setReadOnly(False)
        else:
            self.sbParamsCount.setEnabled(False)
            self.pteEquation.setReadOnly(True)
            for i in range(self.max_param_count):
                self.params_list[i][0].setReadOnly(True)

    def cbPredefModelDepParams_check_changed(self, checked_abbrs):
        if self.cbPredefModelDepParams_changing:
            return

        print('cbPredefModelDepParams_check_changed')
        self.current_model.update_model_options(exp_dep_params=set(checked_abbrs))
        self.pred_setup_field()

    def species_visible_checkbox_toggled(self, checked):
        cb = self.sender()
        print('species_visible_checkbox_toggled', cb.exp_index, cb.text())

        self.current_model.spec_visible[cb.exp_index][cb.text()] = checked
        # dynamically find the corresponding amplitude parameter

        # for i, par in enumerate(self.pred_param_fields['exp_dependent'][cb.exp_index]['params']):
        #     if par.text().startswith(cb.text()):
        #         self.pred_param_fields['exp_dependent'][cb.exp_index]['values'][i].setEnabled(checked)
        #         if not checked:
        #             self.pred_param_fields['exp_dependent'][cb.exp_index]['values'][i].setText('0')
        #         return

    def transfer_param_to_model(self, param_type: str, value):
        """also handles enabling of lower and upper text fields"""
        if self.setting_params:
            return

        par_name = self.sender().par.text()
        if par_name == '':
            return
        if param_type != 'vary':
            try:
                value = float(value)
            except ValueError:
                return
        else:
            # set enabled for lower and upper bound fields
            self.sender().lb.setEnabled(value)
            self.sender().ub.setEnabled(value)

        # print('par name:', par_name, 'type', param_type, 'value', value)
        try:
            self.current_model.params[par_name].__setattr__(param_type, value)
        except KeyError as e:
            print(e.__repr__())

    def _setup_fields(self, fields: dict, params: list):
        for i in range(self.max_param_count):
            visible = len(params) > i

            if visible:
                p = params[i]
                enabled = True
                if hasattr(p, 'enabled'):
                    enabled = p.enabled

                fields['params'][i].setText(p.name)
                fields['lower_bounds'][i].setText(f'{p.min:.4g}')
                fields['lower_bounds'][i].setEnabled(p.vary)
                fields['values'][i].setText(f'{p.value:.4g}')
                fields['values'][i].setEnabled(enabled)
                fields['upper_bounds'][i].setText(f'{p.max:.4g}')
                fields['upper_bounds'][i].setEnabled(p.vary)
                fields['fixed'][i].setChecked(not p.vary)
                fields['fixed'][i].setEnabled(enabled)
                stderr = f'{p.stderr:.4g}' if p.stderr is not None else '0'
                fields['errors'][i].setText(stderr)

            fields['params'][i].setVisible(visible)
            fields['lower_bounds'][i].setVisible(visible)
            fields['values'][i].setVisible(visible)
            fields['upper_bounds'][i].setVisible(visible)
            fields['fixed'][i].setVisible(visible)
            fields['errors'][i].setVisible(visible)

    def pred_setup_field(self):
        if self.current_model is None:
            return

        print('pred_setup_field')
        self.setting_params = True

        self._setup_fields(self.pred_param_fields['exp_independent'], self.current_model.get_model_indep_params())

        model_dep_params = self.current_model.get_model_dep_params_list()
        spec_names = self.current_model.get_current_species_names()
        for i in range(len(self.node)):
            self._setup_fields(self.pred_param_fields['exp_dependent'][i], model_dep_params[i])

            for j in range(self.max_param_count):
                visible = len(spec_names) > j

                if visible:
                    self.pred_spec_visible_fields[i][j].setText(spec_names[j])
                    self.pred_spec_visible_fields[i][j].setChecked(self.current_model.spec_visible[i][spec_names[j]])

                self.pred_spec_visible_fields[i][j].setVisible(visible)

        self.setting_params = False

    def n_spec_changed(self):
        print('n_spec_changed')

        self.cbPredefModelDepParams_changing = True

        self.current_model.update_model_options(n_spec=int(self.sbSpeciesCount.value()))

        av_params = self.current_model.get_available_param_names()  # available parameters

        # set default parameters to checkbox of model dependent parameters
        # check the default values
        self.cbPredefModelDepParams.set_data(av_params)
        default_dep_params = self.current_model.default_exp_dep_params()  # default values
        abbrs = list(self.cbPredefModelDepParams.items.keys())
        for i in range(self.cbPredefModelDepParams.items.__len__()):
            self.cbPredefModelDepParams.set_check_state(i, abbrs[i] in default_dep_params)

        self.cbPredefModelDepParams_changing = False
        self.cbPredefModelDepParams.update_text()

        self.pred_setup_field()

    # def varpro_checked_changed(self, value):
    #     self.current_model.update_model_options(varpro=value)
    #     self.pred_setup_field()

    def model_option_changed(self, value):
        """also handles enabling of lower and upper text fields"""
        if self.setting_params:
            return

        opt_name = self.sender().name

        opts = {opt_name: value}

        self.current_model.update_model_options(**opts)
        self.pred_setup_field()

    def setup_model_options(self):

        opts = self.current_model.model_options()

        # delete all widgets in grid_layout, use walrus operator here
        while w := self.model_options_grid_layout.takeAt(0) is not None:
            self.model_options_grid_layout.removeWidget(w)

        for i, op in enumerate(opts):

            if op['type'] is bool:
                widget = QCheckBox(op['description'])
                widget.setChecked(op['value'])
                widget.toggled.connect(self.model_option_changed)
            else:
                widget = QLineEdit()
                widget.setText(op['value'])
                widget.textChanged.connect(self.model_option_changed)

            widget.name = op['name']  # add param name as attribute to widget

            if op['type'] is bool:
                self.model_options_grid_layout.addWidget(widget, i, 0, 1, 2)
            else:
                self.model_options_grid_layout.addWidget(QtWidgets.QLabel(op['description']), i, 0, 1, 1)
                self.model_options_grid_layout.addWidget(widget, i, 1, 1, 1)

    def model_changed(self):
        # initialize new model
        # self.plotted_fits.clear()
        data = [sp.data for sp in self.node] if self.node is not None else None
        self.current_model = self.models[self.cbModel.currentIndex()](data, n_spec=int(self.sbSpeciesCount.value()))

        self.setup_model_options()

        self.n_spec_changed()

    def clear_plot(self):

        self.fits.clear()
        self.residuals.clear()
        PlotWidget.remove_all_fits()

    def simulate_model(self):

        # self._plot_function()
        self._simulate()
        # try:
        #
        #     self._simulate()
        # except Exception as e:
        #     Logger.message(e.__str__())
        #     QMessageBox.warning(self, 'Plotting Error', e.__str__(), QMessageBox.Ok)

    def fit(self):

        self._fit()
        # try:
        # except Exception as e:
        #     Logger.message(e.__str__())
        #     QMessageBox.warning(self, 'Fitting Error', e.__str__(), QMessageBox.Ok)

    def _simulate(self):
        if self.current_model is None:
            return

        if self.tabWidget.currentIndex() == 1:  # custom model
            return

        try:
            x0, x1 = float(self.leX0.text()), float(self.leX1.text())
        except ValueError:
            QMessageBox.warning(self, "Range", "Fitting range is not valid!", QMessageBox.Ok)
            return

        self.current_model.set_ranges((x0, x1))
        self.current_model.weight_func = self.res_weights[self.cbResWeighting.currentIndex()]['func']

        start_time = time.perf_counter()
        x_vals, fits, residuals = self.current_model.simulate()
        end_time = time.perf_counter()
        print((end_time - start_time) * 1e3, 'ms for simulation')

        if self.current_model.varpro:
            self.pred_setup_field()

        self.plot_fits(x_vals, fits, residuals)

    def _fit(self):

        if self.current_model is None:
            return

        if self.tabWidget.currentIndex() == 1:  # custom model
            return

        try:
            x0, x1 = float(self.leX0.text()), float(self.leX1.text())
        except ValueError:
            QMessageBox.warning(self, "Range", "Fitting range is not valid!", QMessageBox.Ok)
            return

        self.current_model.set_ranges((x0, x1))
        self.current_model.weight_func = self.res_weights[self.cbResWeighting.currentIndex()]['func']

        start_time = time.perf_counter()

        minimizer = Minimizer(self.current_model.residuals, self.current_model.params)

        method = self.methods[self.cbMethod.currentIndex()]['abbr']

        kwds = {'verbose': 2}
        result = minimizer.minimize(method=method, **kwds)  # fit
        self.current_model.params = result.params

        x_vals, fits, residuals = self.current_model.simulate()

        end_time = time.perf_counter()
        print(end_time - start_time, 's for fitting')

        self.pred_setup_field()

        self.plot_fits(x_vals, fits, residuals)

        values_errors = np.zeros((len(result.params), 2), dtype=np.float64)
        for i, p in enumerate(result.params.values()):
            values_errors[i, 0] = p.value
            values_errors[i, 1] = p.stderr if p.stderr is not None else 0

        self.fit_result = FitResult(result, minimizer, values_errors,
                                    self.current_model)

    #
    # def _fit(self):
    #     import numpy as np
    #
    #     x0, x1 = self.lr.getRegion()
    #
    #     start_idx = fi(self.node.data[:, 0], x0)
    #     end_idx = fi(self.node.data[:, 0], x1) + 1
    #
    #     x_data = self.node.data[start_idx:end_idx, 0]
    #     y_data = self.node.data[start_idx:end_idx, 1]
    #
    #     tab_idx = self.tabWidget.currentIndex()
    #
    #     if tab_idx == 0:
    #         self._setup_model()
    #
    #     # fill the parameters from fields
    #     for i, p in enumerate((self.current_model.params if tab_idx == 0 else self.general_model_params).values()):
    #         p.value = float(self.value_list[i][tab_idx].text())
    #         p.min = float(self.lower_bound_list[i][tab_idx].text())
    #         p.max = float(self.upper_bound_list[i][tab_idx].text())
    #         p.vary = not self.fixed_list[i][tab_idx].isChecked()
    #
    #     def y_fit(params):
    #         if tab_idx == 0:
    #             y = self.current_model.wrapper_func(x_data, params)
    #         else:
    #             init, coefs, rates, y0 = self.get_values_from_params(params)
    #             sol = self._simul_custom_model(init, rates, x_data)
    #             y = (coefs * sol).sum(axis=1, keepdims=False) + y0
    #         return y
    #
    #     def residuals(params):
    #         y = y_fit(params)
    #         e = y - y_data
    #         if self.cbPropWeighting.isChecked():
    #             e /= y * y
    #
    #         return e
    #
    #     minimizer = Minimizer(residuals, self.current_model.params if tab_idx == 0 else self.general_model_params)
    #
    #     method = self.methods[self.cbMethod.currentIndex()]['abbr']
    #     result = minimizer.minimize(method=method)  # fit
    #
    #     if tab_idx == 0:
    #         self.current_model.params = result.params
    #     else:
    #         self.general_model_params = result.params
    #
    #     # fill fields
    #     values_errors = np.zeros((len(result.params), 2), dtype=np.float64)
    #     for i, p in enumerate(result.params.values()):
    #         values_errors[i, 0] = p.value
    #         values_errors[i, 1] = p.stderr if p.stderr is not None else 0
    #
    #         self.value_list[i][tab_idx].setText(f"{p.value:.4g}")
    #         self.error_list[i][tab_idx].setText(f"{p.stderr:.4g}" if p.stderr else '')
    #
    #     y_fit_data = y_fit(result.params)
    #     y_residuals = y_data - y_fit_data
    #
    #     # self.remove_last_fit()
    #     self.clear_plot()
    #
    #     self.plot_fit = self.plot_widget.plotItem.plot(x_data, y_fit_data,
    #                                                    pen=pg.mkPen(color=QColor(0, 0, 0, 200), width=2.5),
    #                                                    name="Fit of {}".format(self.node.name))
    #
    #     self.plot_residuals = self.plot_widget.plotItem.plot(x_data, y_residuals,
    #                                                          pen=pg.mkPen(color=QColor(255, 0, 0, 150), width=1),
    #                                                          name="Residuals of {}".format(self.node.name))
    #
    #     self.fitted_spectrum = SpectrumItem.from_xy_values(x_data, y_fit_data,
    #                                                        name="Fit of {}".format(self.node.name),
    #                                                        color='black', line_width=2.5, line_type=Qt.SolidLine)
    #
    #     self.residual_spectrum = SpectrumItem.from_xy_values(x_data, y_residuals,
    #                                                          name="Residuals of {}".format(self.node.name),
    #                                                          color='red', line_width=1, line_type=Qt.SolidLine)
    #
    #     self.fit_result = FitResult(result, minimizer, values_errors, (self.current_model if tab_idx == 0 else self.current_general_model),
    #                                 data_item=self.node, fit_item=self.fitted_spectrum,
    #                                 residuals_item=self.residual_spectrum)

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


    @classmethod
    def replot(cls):
        if cls._instance is not None:
            cls._instance._replot()

    @classmethod
    def clear_plots(cls):
        if cls._instance is not None:
            cls._instance.clear_plot()

    def _replot(self):
        # if no fits simulated, return
        if len(self.fits) != len(self.node):
            return

        pw_plot_items = PlotWidget.plotted_items()
        df = Settings.fit_dark_factor

        for i, (sp_fit, sp_res) in enumerate(zip(self.fits, self.residuals)):
            fit_color = QColor(0, 0, 0, 255)
            res_color = QColor(0, 0, 200, 255)

            if self.node[i] in pw_plot_items:  # set fit_color of checked spectra as darkened
                plot_item = pw_plot_items[self.node[i]]
                node_color = plot_item.opts['pen'].color()
                if not self.cbFitBlackColor.isChecked():
                    fit_color.setRgbF(node_color.redF() * df,
                                      node_color.greenF() * df,
                                      node_color.blueF() * df,
                                      node_color.alphaF())

                    res_color.setRgbF(node_color.redF() * df * 0.9,
                                      node_color.greenF() * df * 0.9,
                                      node_color.blueF() * df * 0.9,
                                      node_color.alphaF())

                PlotWidget.plot_fit(sp_fit, name=f'Fif of {self.node[i].name}',
                                    pen=pg.mkPen(color=fit_color, width=2),
                                    zValue=2e5 - i)

                if self.cbShowResiduals.isChecked():
                    PlotWidget.plot_fit(sp_res, name=f'Residual of {self.node[i].name}',
                                        pen=pg.mkPen(color=res_color, width=1),
                                        zValue=1e5 - i)
                else:
                    PlotWidget.remove_fits([sp_res])
            else:
                # remove them from plot
                PlotWidget.remove_fits([sp_fit, sp_res])

    def plot_fits(self, x_vals, fits, residuals):

        assert len(fits) == len(residuals) == len(x_vals)

        for i in range(len(fits)):
            # only update data if spectra are already instantiated
            if len(self.fits) == len(fits):
                self.fits[i].data = np.vstack((x_vals[i], fits[i])).T
                self.residuals[i].data = np.vstack((x_vals[i], residuals[i])).T
            else:
                sp_fit = SpectrumItem.from_xy_values(x_vals[i], fits[i])
                sp_res = SpectrumItem.from_xy_values(x_vals[i], residuals[i])
                self.fits.append(sp_fit)
                self.residuals.append(sp_res)

        self._replot()

    def _plot_function(self):
        x0, x1 = self.lr.getRegion()

        # if this is a fit widget with real spectrum, use x range from that spectrum,
        # otherwise, use np.linspace with defined number of points in Settings
        if self.node is not None:
            start_idx = fi(self.node.data[:, 0], x0)
            end_idx = fi(self.node.data[:, 0], x1) + 1
            x_data = self.node.data[start_idx:end_idx, 0]
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
                spectra = SpectrumItemGroup(name='all comparments plot')
                for i in range(n_params):
                    name = self.params_list[i+n_params][tab_idx].text()
                    self.plotted_functions.append(self.plot_widget.plotItem.plot(x_data, sol[:, i],
                                                                             pen=pg.mkPen(color=int_default_color_scheme(i),
                                                                                          width=1),
                                                                             name=name))
                    spectra.children.append(SpectrumItem.from_xy_values(x_data, sol[:, i], name=name))
                self.plotted_function_spectra.append(spectra)
                return

            y_data = (sol * coefs).sum(axis=1, keepdims=False) + y0

        params = ', '.join([f"{self.params_list[i][tab_idx].text()}={self.value_list[i][tab_idx].text()}" for i in range(par_count)])

        name = f"Func plot: {params}"
        self.plotted_functions.append(self.plot_widget.plotItem.plot(x_data, y_data,
                                                                     pen=pg.mkPen(color=QColor(0, 0, 0, 255),
                                                                                  width=1),
                                                                     name=name))

        self.plotted_function_spectra.append(SpectrumItem.from_xy_values(x_data, y_data, name=name))

    def set_lower_upper_enabled(self):
        if self.node is None:
            return
        for i in range(len(self.fixed_list)):
            enabled_0 = not self.fixed_list[i][0].isChecked()
            enabled_1 = not self.fixed_list[i][1].isChecked()

            self.lower_bound_list[i][0].setEnabled(enabled_0)
            self.upper_bound_list[i][0].setEnabled(enabled_0)

            self.lower_bound_list[i][1].setEnabled(enabled_1)
            self.upper_bound_list[i][1].setEnabled(enabled_1)

    def fixed_checked_changed(self, value):
        if self.node is None:
            return

        checkbox = self.sender()  # get the checkbox that was clicked on

        idx = self.tabWidget.currentIndex()

        i = 0
        for i, (ch1, ch2) in enumerate(self.fixed_list):
            if ch1 == checkbox or ch2 == checkbox:
                break

        enabled = True if value == Qt.Unchecked else False

        self.lower_bound_list[i][idx].setEnabled(enabled)
        self.upper_bound_list[i][idx].setEnabled(enabled)

    def params_count_changed(self):

        count = int(self.sbParamsCount.value())

        self.set_field_visible(count, 0)

    def set_field_visible(self, n_params, idx=0):
        for i in range(self.max_param_count):
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
            report = f"Fitting of '{self.node[0].parent.name if len(self.node) > 0 else self.node[0].name}' succeeded. " \
                     f"You can find fit results in variable 'fit'." \
                     f"\n{self.fit_result.report(print=False)}\n"

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
        # self.plot_widget.removeItem(self.lr)
        # del self.lr
        PlotWidget.remove_linear_region()
        self.dock_widget.setVisible(False)
        self.accepted_func()
        # super(FitWidget, self).accept()
        PlotWidget.remove_all_fits()


    def reject(self):
        # self.remove_last_fit()
        self.clear_plot()
        FitWidget.is_opened = False
        FitWidget._instance = None
        # self.plot_widget.removeItem(self.lr)
        # del self.lr
        PlotWidget.remove_linear_region()
        self.dock_widget.setVisible(False)
        PlotWidget.remove_all_fits()


        # super(FitWidget, self).reject()


if __name__ == "__main__":
    import sys

    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    app = QtWidgets.QApplication(sys.argv)
    Dialog = FitWidget(None, None)
    # Dialog.show()
    sys.exit(app.exec_())
