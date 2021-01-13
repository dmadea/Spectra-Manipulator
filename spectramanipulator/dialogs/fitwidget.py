from PyQt5 import QtWidgets

from PyQt5.QtCore import Qt
from .gui_fit_widget import Ui_Form

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QCheckBox, QFileDialog
import numpy as np

from .checkbox_rc import CheckBoxRC

from spectramanipulator.settings import Settings
from spectramanipulator.logger import Logger
from spectramanipulator.treeview.item import SpectrumItem, SpectrumItemGroup
from spectramanipulator.console import Console
from spectramanipulator.spectrum import fi
from spectramanipulator.dialogs.trust_reg_refl_option_dialog import TrustRegionReflOptionDialog
import spectramanipulator
from .param_settings_dialog import ParamSettingsDialog

import pyqtgraph as pg

# from user_namespace import UserNamespace

from lmfit import Minimizer, Parameters
from scipy.integrate import odeint

import spectramanipulator.fitting.fitmodels as fitmodels

import inspect
from ..general_model import GeneralModel

from ..plotwidget import PlotWidget
from .fitresult import FitResult

from ..utils.syntax_highlighter import KineticModelHighlighter

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
    max_param_count = 45

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

        self.lr = None  # linear region item
        self.show_region_checked_changed()

        # self.current_model = None  # actual model, it could be general model or predefined model
        self._general_model = None
        self.saving_general_model = False

        self.cbVarProAmps.name = "varpro"
        self.cbVarProAmps.toggled.connect(self.model_option_changed)
        self.cbVarProIntercept.name = "fit_intercept_varpro"
        self.cbVarProIntercept.toggled.connect(self.model_option_changed)

        self._predefined_model = None
        self.general_model_params = None
        self.fit_result = None

        self.fits = SpectrumItemGroup(name='Fits')
        self.residuals = SpectrumItemGroup(name='Residuals')

        # update region when the focus is lost and when the user presses enter
        self.leX0.returnPressed.connect(self.update_region)
        self.leX1.returnPressed.connect(self.update_region)
        self.leX0.focus_lost.connect(self.update_region)
        self.leX1.focus_lost.connect(self.update_region)

        self.update_region_text_values()

        self.btnFit.clicked.connect(self.fit)
        self.btnSimulateModel.clicked.connect(self.simulate_model)
        self.btnPrintReport.clicked.connect(self.print_report)
        self.btnClearPlot.clicked.connect(self.clear_plot)
        self.btnOK.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)
        self.cbGenModels.currentIndexChanged.connect(self.cbGenModel_changed)
        self.btnBuildModel.clicked.connect(self.build_gen_model)
        self.cbShowBackwardRates.stateChanged.connect(self.cbShowBackwardRates_checked_changed)
        self.btnSaveCustomModel.clicked.connect(self.save_general_model_as)
        self.sbSpeciesCount.valueChanged.connect(lambda: self.n_spec_changed())
        self.cbShowResiduals.toggled.connect(self._replot)
        self.cbFitBlackColor.toggled.connect(self._replot)
        self.cbShowRegion.toggled.connect(self.show_region_checked_changed)
        self.tbAlgorithmSettings.clicked.connect(self.tbAlgorithmSettings_clicked)

        # predefined kinetic model setup

        self.ppteScheme_highlighter = KineticModelHighlighter(self.pteScheme.document())

        self.model_options_grid_layout = QtWidgets.QGridLayout(self)
        self.verticalLayout.addLayout(self.model_options_grid_layout)
        self.pred_model_indep_params_layout = self.create_params_layout()
        self.general_model_indep_params_layout = self.create_params_layout()
        self.verticalLayout.addLayout(self.pred_model_indep_params_layout)
        self.verticalLayout_2.addLayout(self.general_model_indep_params_layout)

        self.cbPredefModelDepParams = ComboBoxCB(self)
        self.cbPredefModelDepParams_changing = False
        self.cbPredefModelDepParams.check_changed.connect(self.cbPredefModelDepParams_check_changed)

        self.cbGeneralModelDepParams = ComboBoxCB(self)
        self.cbGeneralModelDepParams_changing = False
        self.cbGeneralModelDepParams.check_changed.connect(self.cbGeneralModelDepParams_check_changed)

        hbox = QtWidgets.QHBoxLayout(self)
        hbox2 = QtWidgets.QHBoxLayout(self)
        hbox.addWidget(QtWidgets.QLabel('Selected experiment-dependent parameters:'))
        hbox2.addWidget(QtWidgets.QLabel('Selected experiment-dependent parameters:'))
        hbox.addWidget(self.cbPredefModelDepParams)
        hbox2.addWidget(self.cbGeneralModelDepParams)
        btnExperimentParamsSettings_predefined = QtWidgets.QToolButton()
        btnExperimentParamsSettings_genearal = QtWidgets.QToolButton()
        btnExperimentParamsSettings_predefined.setText('...')
        btnExperimentParamsSettings_genearal.setText('...')
        btnExperimentParamsSettings_predefined.clicked.connect(self.exp_dep_param_sett_clicked)
        btnExperimentParamsSettings_genearal.clicked.connect(self.exp_dep_param_sett_clicked)
        hbox.addWidget(btnExperimentParamsSettings_predefined)
        hbox2.addWidget(btnExperimentParamsSettings_genearal)
        self.verticalLayout.addLayout(hbox)
        self.verticalLayout_2.addLayout(hbox2)

        self.tab_widget_pred_model = QtWidgets.QTabWidget(self)
        self.verticalLayout.addWidget(self.tab_widget_pred_model)

        self.tab_widget_general_model = QtWidgets.QTabWidget(self)
        self.verticalLayout_2.addWidget(self.tab_widget_general_model)

        self.pred_species_hlayouts = []
        self.pred_model_dep_param_layouts = []

        self.general_species_hlayouts = []
        self.general_model_dep_param_layouts = []

        if self.node is not None:
            for spectrum in self.node:
                widget, species_hlayout, params_layout = self.create_tab_widget(self.tab_widget_pred_model)
                self.pred_species_hlayouts.append(species_hlayout)
                self.pred_model_dep_param_layouts.append(params_layout)

                widget_2, species_hlayout_2, params_layout_2 = self.create_tab_widget(self.tab_widget_general_model)
                self.general_species_hlayouts.append(species_hlayout_2)
                self.general_model_dep_param_layouts.append(params_layout_2)

                self.tab_widget_pred_model.addTab(widget, spectrum.name)
                self.tab_widget_general_model.addTab(widget_2, spectrum.name)

        vals_temp = dict(params=[], lower_bounds=[], values=[], upper_bounds=[], fixed=[], errors=[])
        self.pred_param_fields = dict(exp_independent=deepcopy(vals_temp),
                                      exp_dependent=[deepcopy(vals_temp) for _ in range(len(self.node))])

        self.pred_spec_visible_fields = [[] for _ in range(len(self.node))]
        self.general_param_fields = deepcopy(self.pred_param_fields)
        self.general_spec_visible_fields = deepcopy(self.pred_spec_visible_fields)

        def add_widgets(dict_fields: dict, param_layout: QtWidgets.QGridLayout):
            for i in range(self.max_param_count):
                par = QLineEdit()  # parameter text field
                lb = QLineEdit()  # lower bound text field
                val = QLineEdit()  # value text field
                ub = QLineEdit()  # upper bound text field
                fix = CheckBoxRC()  # fixed checkbox
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
            for _ in range(self.max_param_count):
                for key in dict_fields.keys():
                    for first, second in zip(dict_fields[key][:-1], dict_fields[key][1:]):
                        self.setTabOrder(first, second)

        add_widgets(self.pred_param_fields['exp_independent'], self.pred_model_indep_params_layout)
        set_tab_order(self.pred_param_fields['exp_independent'])

        add_widgets(self.general_param_fields['exp_independent'], self.general_model_indep_params_layout)
        set_tab_order(self.general_param_fields['exp_independent'])

        for i in range(len(self.node)):
            add_widgets(self.pred_param_fields['exp_dependent'][i], self.pred_model_dep_param_layouts[i])
            set_tab_order(self.pred_param_fields['exp_dependent'][i])

            add_widgets(self.general_param_fields['exp_dependent'][i], self.general_model_dep_param_layouts[i])
            set_tab_order(self.general_param_fields['exp_dependent'][i])

            # setup species visible checkboxes
            for _ in range(self.max_param_count):
                cb_predefined_model = CheckBoxRC('')
                cb_predefined_model.exp_index = i
                cb_predefined_model.setVisible(False)
                cb_general_model = CheckBoxRC('')
                cb_general_model.exp_index = i
                cb_general_model.setVisible(False)

                cb_predefined_model.toggled.connect(self.species_visible_checkbox_toggled)
                cb_general_model.toggled.connect(self.species_visible_checkbox_toggled)

                self.pred_spec_visible_fields[i].append(cb_predefined_model)
                self.pred_species_hlayouts[i].addWidget(cb_predefined_model)

                self.general_spec_visible_fields[i].append(cb_general_model)
                self.general_species_hlayouts[i].addWidget(cb_general_model)
            self.pred_species_hlayouts[i].addSpacerItem(QtWidgets.QSpacerItem(1, 1,
                                                        QtWidgets.QSizePolicy.Expanding,
                                                        QtWidgets.QSizePolicy.Fixed))
            self.general_species_hlayouts[i].addSpacerItem(QtWidgets.QSpacerItem(1, 1,
                                                           QtWidgets.QSizePolicy.Expanding,
                                                           QtWidgets.QSizePolicy.Fixed))

        self.model_option_widgets = []  # only for predefined model

        self.setting_params = False

        # get all models from fitmodels, get classes that inherits from Model base class and sort them by name
        # and number of species
        classes = inspect.getmembers(sys.modules[fitmodels.__name__], inspect.isclass)
        tuples = filter(lambda tup: issubclass(tup[1], fitmodels.Model) and
                        (tup[1] is not fitmodels.Model and tup[1] is not fitmodels.GeneralFitModel), classes)
        # load models
        self.models = sorted(list(map(lambda tup: tup[1], tuples)), key=lambda cls: cls.name)
        # fill the available models combo box with model names
        self.cbModel.addItems(map(lambda m: m.name, self.models))

        self.cbModel.currentIndexChanged.connect(self.predefined_model_changed)

        self.gen_models_paths = []
        self.update_general_models()

        self.res_weights = [
            {'description': 'No weighting', 'func': lambda res, y: res},
            {'description': 'Absorbance weighting: noise \u221d 10 ^ y', 'func': lambda res, y: res * np.exp(-y * np.log(10))},
            {'description': 'Proportional weighting: noise \u221d y', 'func': lambda res, y: res / y},
        ]

        self.cbResWeighting.addItems(map(lambda m: m['description'], self.res_weights))

        # abbr is name that is needed for lmfit.fitting method
        self.methods = [
            {'name': 'Trust Region Reflective method', 'abbr': 'least_squares', 'option_dialog': TrustRegionReflOptionDialog},
            {'name': 'Levenberg–Marquardt', 'abbr': 'leastsq', 'option_dialog': None},
            {'name': 'Nelder-Mead, Simplex method (no error)', 'abbr': 'nelder', 'option_dialog': None},
            {'name': 'Differential evolution', 'abbr': 'differential_evolution', 'option_dialog': None},
            {'name': 'L-BFGS-B (no error)', 'abbr': 'lbfgsb', 'option_dialog': None},
            {'name': 'Powell (no error)', 'abbr': 'powell', 'option_dialog': None}
        ]

        self.cbMethod.addItems(map(lambda m: m['name'], self.methods))
        self.cbMethod.currentIndexChanged.connect(self.cbMethod_currentIndexChanged)
        self.cbMethod_currentIndexChanged()
        self.model_options_dict = {}

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

        self.setup_general_model()
        self.predefined_model_changed()

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
        if self._general_model is None:
            return

        curr_model_path = self.gen_models_paths[self.cbGenModels.currentIndex()]
        file = "Json (*.json)"

        filepath = QFileDialog.getSaveFileName(caption="Save Custom Kinetic Model",
                                               directory=curr_model_path,
                                               filter=file, initialFilter=file)
        if filepath[0] == '':
            return

        try:
            self._general_model.scheme = self.pteScheme.toPlainText()
            j = np.asarray([self._general_model.params[p].value for p in self._general_model.param_names_dict[0]['j']])
            rates = self._general_model.get_rate_values(0)

            self._general_model.set_rates(rates)
            self._general_model.initial_conditions = dict(zip(self._general_model.get_compartments(), j))
            self._general_model.save(filepath[0])
        except Exception as e:
            Logger.message(e.__str__())
            QMessageBox.critical(self, 'Saving error', 'Error, first build the model.', QMessageBox.Ok)
            return

        self.saving_general_model = True
        self.update_general_models()
        # set saved model as current index
        for k, fpath in enumerate(self.gen_models_paths):
            fname = os.path.splitext(os.path.split(fpath)[1])[0]
            if fname == os.path.splitext(os.path.split(filepath[0])[1])[0]:
                self.cbGenModels.setCurrentIndex(k)
                break

        self.saving_general_model = False

    def cbGenModel_changed(self):
        if self._general_model is None or self.saving_general_model:
            return

        self._general_model.load_from_file(self.gen_models_paths[self.cbGenModels.currentIndex()])
        self.pteScheme.setPlainText(self._general_model.general_model.scheme)
        self.general_model_setup_dep_params()
        self.setup_fields(self._general_model)

    def cbShowBackwardRates_checked_changed(self):
        if self._general_model is None:
            return

        self._general_model.update_model_options(show_backward_rates=self.cbShowBackwardRates.isChecked())
        self.general_model_setup_dep_params()
        self.setup_fields(self._general_model)

    def build_gen_model(self):
        self._general_model.load_from_scheme(self.pteScheme.toPlainText())
        self.general_model_setup_dep_params()
        self.setup_fields(self._general_model)

    def cbGeneralModelDepParams_check_changed(self, checked_abbrs):
        if self.cbGeneralModelDepParams_changing:
            return
        Logger.debug('cbGeneralModelDepParams_check_changed')

        self._general_model.update_model_options(exp_dep_params=checked_abbrs)
        self.setup_fields(self._general_model)

    def cbPredefModelDepParams_check_changed(self, checked_abbrs):
        if self.cbPredefModelDepParams_changing:
            return
        Logger.debug('cbPredefModelDepParams_check_changed')

        self._predefined_model.update_model_options(exp_dep_params=set(checked_abbrs))
        self.setup_fields(self._predefined_model)

    def species_visible_checkbox_toggled(self, checked):
        model = self._predefined_model if self.tabWidget.currentIndex() == 0 else self._general_model

        if model is None or self.setting_params:
            return

        cb = self.sender()
        Logger.debug('species_visible_checkbox_toggled', cb.exp_index, cb.text())

        if cb.right_button_pressed:
            model.set_all_spec_visible(checked, cb.text())
        else:
            model.spec_visible[cb.exp_index][cb.text()] = checked

        model.update_model_options()
        self.setup_fields()

    def transfer_param_to_model(self, param_type: str, value):
        """also handles enabling of lower and upper text fields"""
        if self.setting_params:
            return

        model = self._predefined_model if self.tabWidget.currentIndex() == 0 else self._general_model

        par_name = self.sender().par.text()
        if par_name == '':
            return
        if param_type != 'vary':
            try:
                value = float(value)
            except ValueError:
                return
        else:  # handles fix toggle
            # set enabled for lower and upper bound fields
            self.sender().lb.setEnabled(value)
            self.sender().ub.setEnabled(value)

            # if right mouse button was pressed, fix/unfix all experiments params
            if self.sender().right_button_pressed:
                model.fix_all_exp_dep_params(value, par_name)
                self.setup_fields()
                return

        # print('par name:', par_name, 'type', param_type, 'value', value)
        try:
            model.params[par_name].__setattr__(param_type, value)
        except KeyError as e:
            print(e.__repr__())

    def _fill_fields(self, fields: dict, params: list):
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

    def setup_fields(self, model=None):
        """Transfer parameters from model to all fields"""
        if model is None:
            model = self._predefined_model if self.tabWidget.currentIndex() == 0 else self._general_model

        Logger.debug('setup_field')
        self.setting_params = True  # prevent calling transfer_param_to_model and species_visible_checkbox_toggled

        if model == self._predefined_model:
            param_fields = self.pred_param_fields
            vis_fields = self.pred_spec_visible_fields
        else:
            param_fields = self.general_param_fields
            vis_fields = self.general_spec_visible_fields

        self._fill_fields(param_fields['exp_independent'], model.get_model_indep_params_list())

        model_dep_params = model.get_model_dep_params_list()
        spec_names = model.get_current_species_names()
        for i in range(len(self.node)):
            self._fill_fields(param_fields['exp_dependent'][i], model_dep_params[i])

            for j in range(self.max_param_count):
                visible = len(spec_names) > j

                if visible:
                    vis_fields[i][j].setText(spec_names[j])
                    vis_fields[i][j].setChecked(model.spec_visible[i][spec_names[j]])

                vis_fields[i][j].setVisible(visible)

        self.setting_params = False

    def exp_dep_param_sett_clicked(self):
        model = self._predefined_model if self.tabWidget.currentIndex() == 0 else self._general_model

        if model is None or self.setting_params:
            return

        param_fields = self.pred_param_fields if model == self._predefined_model else self.general_param_fields
        par_names = list(model.exp_dep_params)

        exp_dep_params = []

        for i in range(len(par_names)):  # number of parameters
            lst_vals = []
            d = dict(name=par_names[i], values=lst_vals)
            for j in range(len(self.node)):  # number of experiments
                lst_vals.append(param_fields['exp_dependent'][j]['values'][i].text())
            exp_dep_params.append(d)

        def set_result():
            for i in range(len(psd.exp_dep_params)):  # number of parameters
                for j in range(len(self.node)):  # number of experiments
                    par_value = psd.exp_dep_params[i]['values'][j]
                    par_name = param_fields['exp_dependent'][j]['params'][i].text()

                    try:
                        model.params[par_name].value = par_value
                    except KeyError as e:
                        print(e.__repr__())

            self.setup_fields()  # update fields

        psd = ParamSettingsDialog(exp_dep_params, set_result=set_result)
        psd.show()

    def general_model_setup_dep_params(self):
        self.cbGeneralModelDepParams_changing = True

        av_params = self._general_model.get_all_param_names()  # available parameters

        # set default parameters to checkbox of model dependent parameters
        # check the default values
        self.cbGeneralModelDepParams.set_data(av_params)
        abbrs = list(self.cbGeneralModelDepParams.items.keys())
        for i in range(self.cbGeneralModelDepParams.items.__len__()):
            self.cbGeneralModelDepParams.set_check_state(i, abbrs[i] in self._general_model.exp_dep_params)

        self.cbGeneralModelDepParams_changing = False
        # self.cbGeneralModelDepParams_check_changed()
        self.cbGeneralModelDepParams.update_text()

    def n_spec_changed(self):
        Logger.debug('n_spec_changed')

        self.cbPredefModelDepParams_changing = True

        self._predefined_model.update_model_options(n_spec=int(self.sbSpeciesCount.value()))

        av_params = self._predefined_model.get_all_param_names()  # available parameters
        default_params = self._predefined_model.default_exp_dep_params()

        # set default parameters to checkbox of model dependent parameters
        # check the default values
        self.cbPredefModelDepParams.set_data(av_params)
        # default_dep_params = self._predefined_model.default_exp_dep_params()  # default values
        abbrs = list(self.cbPredefModelDepParams.items.keys())
        for i in range(self.cbPredefModelDepParams.items.__len__()):
            self.cbPredefModelDepParams.set_check_state(i, abbrs[i] in default_params)

        self.cbPredefModelDepParams_changing = False
        self.cbPredefModelDepParams.update_text()

        self.setup_fields(self._predefined_model)

    def model_option_changed(self, value):
        """also handles enabling of lower and upper text fields"""
        if self.setting_params:
            return

        model = self._predefined_model if self.tabWidget.currentIndex() == 0 else self._general_model

        opt_name = self.sender().name

        opts = {opt_name: value}

        model.update_model_options(**opts)
        self.setup_fields()

    def setup_general_model(self):
        data = [sp.data for sp in self.node] if self.node is not None else None
        self._general_model = fitmodels.GeneralFitModel(data, varpro=self.cbVarProAmps.isChecked(),
                                                        fit_intercept_varpro=self.cbVarProIntercept.isChecked())
        self.cbGenModel_changed()

    def predefined_model_changed(self):
        # initialize new model
        data = [sp.data for sp in self.node] if self.node is not None else None
        self._predefined_model = self.models[self.cbModel.currentIndex()](data, n_spec=int(self.sbSpeciesCount.value()))

        self.setup_pred_model_options()

        self.n_spec_changed()

    def setup_pred_model_options(self):
        """Add option fields that are associated with the selected model"""

        opts = self._predefined_model.model_options()

        # delete all widgets in grid_layout, use walrus operator here
        for w in self.model_option_widgets:
            self.model_options_grid_layout.removeWidget(w)  # remove widget from layout
            w.deleteLater()  # delete the widget https://stackoverflow.com/questions/10716300/removing-qwidgets-from-a-qgridlayout

        self.model_option_widgets.clear()

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

            self.model_option_widgets.append(widget)

            if op['type'] is bool:
                self.model_options_grid_layout.addWidget(widget, i, 0, 1, 2)
            else:
                qlabel = QtWidgets.QLabel(op['description'])
                self.model_option_widgets.append(qlabel)
                self.model_options_grid_layout.addWidget(qlabel, i, 0, 1, 1)
                self.model_options_grid_layout.addWidget(widget, i, 1, 1, 1)

    def clear_plot(self):
        self.fits.children.clear()
        self.residuals.children.clear()
        PlotWidget.remove_all_fits()

    def cbMethod_currentIndexChanged(self):
        dialog_cls = self.methods[self.cbMethod.currentIndex()]['option_dialog']
        if dialog_cls is None:
            self.model_options_dict = {}
            return

        self.model_options_dict = dialog_cls.default_opts()

    def tbAlgorithmSettings_clicked(self):
        def set_result():
            self.model_options_dict = dialog.options

        dialog_cls = self.methods[self.cbMethod.currentIndex()]['option_dialog']

        if dialog_cls is None:
            return

        dialog = dialog_cls(set_result=set_result, **self.model_options_dict)
        dialog.show()

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
        model = self._predefined_model if self.tabWidget.currentIndex() == 0 else self._general_model

        if model is None:
            return

        try:
            x0, x1 = float(self.leX0.text()), float(self.leX1.text())
        except ValueError:
            QMessageBox.warning(self, "Range", "Fitting range is not valid!", QMessageBox.Ok)
            return

        model.set_ranges((x0, x1))
        model.weight_func = self.res_weights[self.cbResWeighting.currentIndex()]['func']

        start_time = time.perf_counter()
        x_vals, fits, residuals = model.simulate()
        end_time = time.perf_counter()
        Logger.debug((end_time - start_time) * 1e3, 'ms for simulation')

        if model.varpro:
            self.setup_fields()

        self.plot_fits(x_vals, fits, residuals)

    def _fit(self):
        model = self._predefined_model if self.tabWidget.currentIndex() == 0 else self._general_model

        if model is None:
            return

        try:
            x0, x1 = float(self.leX0.text()), float(self.leX1.text())
        except ValueError:
            QMessageBox.warning(self, "Range", "Fitting range is not valid!", QMessageBox.Ok)
            return

        model.set_ranges((x0, x1))
        model.weight_func = self.res_weights[self.cbResWeighting.currentIndex()]['func']

        start_time = time.perf_counter()

        minimizer = Minimizer(model.residuals, model.params)
        method = self.methods[self.cbMethod.currentIndex()]['abbr']

        result = minimizer.minimize(method=method, **self.model_options_dict)  # fit

        end_time = time.perf_counter()
        Logger.debug(end_time - start_time, 's for fitting')

        values_errors = np.zeros((len(result.params), 2), dtype=np.float64)
        for i, (p, new_p) in enumerate(zip(model.params.values(), result.params.values())):
            p.value = new_p.value  # update fitted parameters
            values_errors[i, 0] = p.value
            values_errors[i, 1] = p.stderr if p.stderr is not None else 0

        x_vals, fits, residuals = model.simulate()

        self.setup_fields()
        self.plot_fits(x_vals, fits, residuals)

        self.fit_result = FitResult(result, minimizer, values_errors,
                                    model)

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
    #

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
                sp_fit = SpectrumItem.from_xy_values(x_vals[i], fits[i], name=f'Fif of {self.node[i].name}')
                sp_res = SpectrumItem.from_xy_values(x_vals[i], residuals[i], name=f'Residual of {self.node[i].name}')
                self.fits.children.append(sp_fit)
                self.residuals.children.append(sp_res)

        self._replot()

    # def _plot_function(self):
    #     x0, x1 = self.lr.getRegion()
    #
    #     # if this is a fit widget with real spectrum, use x range from that spectrum,
    #     # otherwise, use np.linspace with defined number of points in Settings
    #     if self.node is not None:
    #         start_idx = fi(self.node.data[:, 0], x0)
    #         end_idx = fi(self.node.data[:, 0], x1) + 1
    #         x_data = self.node.data[start_idx:end_idx, 0]
    #     else:
    #         x_data = np.linspace(x0, x1, num=Settings.FP_num_of_points)
    #
    #     tab_idx = self.tabWidget.currentIndex()
    #
    #     if tab_idx == 0:  # equation-based model
    #         self._setup_model()
    #
    #         par_count = self.current_model.par_count()
    #
    #         # get params from field to current model
    #         for i in range(par_count):
    #             param = self.params_list[i][0].text()
    #             self.current_model.params[param].value = float(self.value_list[i][0].text())
    #
    #         # calculate the y values according to our model
    #         y_data = self.current_model.wrapper_func(x_data, self.current_model.params)
    #     else:  # custom kinetic model
    #         if self.current_general_model is None:
    #             self.cbGenModel_changed()
    #
    #         init, coefs, rates, y0 = self.get_params_from_fields()
    #         par_count = 2 * init.shape[0] + rates.shape[0]
    #         n_params = len(init)
    #
    #         sol = self._simul_custom_model(init, rates, x_data)
    #
    #         if self.cbPlotAllComps.isChecked():
    #             spectra = SpectrumItemGroup(name='all comparments plot')
    #             for i in range(n_params):
    #                 name = self.params_list[i+n_params][tab_idx].text()
    #                 self.plotted_functions.append(self.plot_widget.plotItem.plot(x_data, sol[:, i],
    #                                                                          pen=pg.mkPen(color=int_default_color_scheme(i),
    #                                                                                       width=1),
    #                                                                          name=name))
    #                 spectra.children.append(SpectrumItem.from_xy_values(x_data, sol[:, i], name=name))
    #             self.plotted_function_spectra.append(spectra)
    #             return
    #
    #         y_data = (sol * coefs).sum(axis=1, keepdims=False) + y0
    #
    #     params = ', '.join([f"{self.params_list[i][tab_idx].text()}={self.value_list[i][tab_idx].text()}" for i in range(par_count)])
    #
    #     name = f"Func plot: {params}"
    #     self.plotted_functions.append(self.plot_widget.plotItem.plot(x_data, y_data,
    #                                                                  pen=pg.mkPen(color=QColor(0, 0, 0, 255),
    #                                                                               width=1),
    #                                                                  name=name))
    #
    #     self.plotted_function_spectra.append(SpectrumItem.from_xy_values(x_data, y_data, name=name))

    # def set_lower_upper_enabled(self):
    #     if self.node is None:
    #         return
    #     for i in range(len(self.fixed_list)):
    #         enabled_0 = not self.fixed_list[i][0].isChecked()
    #         enabled_1 = not self.fixed_list[i][1].isChecked()
    #
    #         self.lower_bound_list[i][0].setEnabled(enabled_0)
    #         self.upper_bound_list[i][0].setEnabled(enabled_0)
    #
    #         self.lower_bound_list[i][1].setEnabled(enabled_1)
    #         self.upper_bound_list[i][1].setEnabled(enabled_1)

    # def fixed_checked_changed(self, value):
    #     if self.node is None:
    #         return
    #
    #     checkbox = self.sender()  # get the checkbox that was clicked on
    #
    #     idx = self.tabWidget.currentIndex()
    #
    #     i = 0
    #     for i, (ch1, ch2) in enumerate(self.fixed_list):
    #         if ch1 == checkbox or ch2 == checkbox:
    #             break
    #
    #     enabled = True if value == Qt.Unchecked else False
    #
    #     self.lower_bound_list[i][idx].setEnabled(enabled)
    #     self.upper_bound_list[i][idx].setEnabled(enabled)

    # def params_count_changed(self):
    #
    #     count = int(self.sbParamsCount.value())
    #
    #     self.set_field_visible(count, 0)

    # def set_field_visible(self, n_params, idx=0):
    #     for i in range(self.max_param_count):
    #         visible = n_params > i
    #
    #         self.params_list[i][idx].setVisible(visible)
    #         self.lower_bound_list[i][idx].setVisible(visible)
    #         self.value_list[i][idx].setVisible(visible)
    #         self.upper_bound_list[i][idx].setVisible(visible)
    #         self.fixed_list[i][idx].setVisible(visible)
    #         self.error_list[i][idx].setVisible(visible)

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
        self.accepted_func()

        self.clear_plot()
        PlotWidget.remove_linear_region()

        self.print_report()

        self.accepted = True
        self.is_opened = False
        self._instance = None
        self.dock_widget.setVisible(False)

    def reject(self):
        self.clear_plot()
        PlotWidget.remove_linear_region()
        self.is_opened = False
        self._instance = None
        self.dock_widget.setVisible(False)


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
