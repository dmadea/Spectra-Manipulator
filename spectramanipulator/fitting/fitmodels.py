# from abc import abstractmethod
from lmfit import Parameters
import numpy as np
import inspect
from scipy.integrate import odeint
from ..spectrum import fi
import scipy
from copy import deepcopy
from scipy.linalg import lstsq
from ..general_model import GeneralModel

posv = scipy.linalg.get_lapack_funcs(('posv'))
# gels = scipy.linalg.get_lapack_funcs(('gels'))


# import numba as nb

# import matplotlib.pyplot as plt
# from sympy import Derivative, symbols, Function, Equality
# from console import Console

def get_xy(data, x0=None, x1=None):
    """Get x and y values for current exp data for range [x0, x1]."""
    start = 0
    end = data.shape[0]

    if x0 is not None and x1 is not None and x0 > x1:
        x0, x1 = x1, x0

    if x0 is not None:
        start = fi(data[:, 0], x0)

    if x1 is not None:
        end = fi(data[:, 0], x1) + 1

    x = data[start:end, 0]
    y = data[start:end, 1]

    return x, y

def OLS_ridge(A, B, alpha=0.000):
    """fast solve least squares solution for X: AX=B by ordinary least squares, with direct solve,
    with optional Tikhonov regularization"""

    ATA = A.T.dot(A)
    ATB = A.T.dot(B)

    if alpha != 0:
        ATA.flat[::ATA.shape[-1] + 1] += alpha

    # call the LAPACK function, direct solve method
    c, x, info = posv(ATA, ATB, lower=False,
                      overwrite_a=False,
                      overwrite_b=False)

    return x

# def lstsq_fast(A, b):
#     # TODO---->>
#
#     return lstsq(A, b, lapack_driver='gelss')[0]

# @vectorize(nopython=True)
# def fold_exp(t, k, fwhm):
#
#     w = fwhm / (2 * np.sqrt(np.log(2)))  # width
#
#     if w > 0:
#         return 0.5 * np.exp(k * (k * w * w / 4.0 - t)) * math_erfc(w * k / 2.0 - t / w)
#     else:
#         return np.exp(-t * k) if t >= 0 else 0

def target_1st_order(t, K, j, numerical=False):
    """ t - times vector,
        K - Transfer matrix, j - initial population vector"""

    if numerical:
        # return odeint(lambda c, _t: K.dot(c), j, t)
        raise NotImplementedError()

    # based on Ivo H.M. van Stokkum equation in doi:10.1016/j.bbabio.2004.04.011

    L, Q = np.linalg.eig(K)
    Q_inv = np.linalg.inv(Q)

    A2_T = Q * Q_inv.dot(j)  # Q @ np.diag(Q_inv.dot(j))

    C = np.heaviside(t[:, None], 1) * np.exp(t[:, None] * L[None, :])

    return C.dot(A2_T.T)

# abstract class, all models has to inherit from this class
class Model(object):

    name = '--model name--'

    def __init__(self, exps_data: list, ranges: [list, tuple] = None, varpro: bool = True,
                 exp_dep_params: list = None, n_spec: int = 2,
                 spec_names: [list, tuple] = None, spec_visible: dict = None,
                 weight_func=lambda res, y: res, **kwargs):
        """ TODO-->> exp_data is list of 2D ndaarrays """
        self.exps_data = list(exps_data) if isinstance(exps_data, (list, tuple)) else [exps_data]
        self.ranges = ranges  # list of lists or None for no ranges
        self.set_ranges(self.ranges)

        self.varpro = varpro  # use Variable Projection - efficient calculation of amplitudes
        self.exp_dep_params = exp_dep_params  # experiment dependent params
        self.n_spec = n_spec
        self.params = None
        self.weight_func = weight_func

        self.spec_names = spec_names
        if self.spec_names is None:
            self.spec_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        self.spec_visible = spec_visible  # dictionary
        self.exp_indep_params = None  # experimental independent parameters, will be set in init_params method

        if self.spec_visible is None:
            self.spec_visible = [{name: True for name in self.spec_names[:self.n_spec]} for _ in range(len(self.exps_data))]

        self.param_names_dict = {}

    def set_ranges(self, ranges=None):
        if self.exps_data is None:
            return

        if ranges is None:
            self.ranges = [(None, None)] * len(self.exps_data)
            return

        # only one range, make it valid for all experiements
        if isinstance(ranges, (list, tuple)) and isinstance(ranges[0], (float, int)):
            self.ranges = [ranges] * len(self.exps_data)
        else:
            self.ranges = ranges

    def get_current_species_names(self):
        return self.spec_names[:self.n_spec]

    def update_model_options(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f'Argument {key} is not valid.')
            self.__setattr__(key, value)

        if 'n_spec' in kwargs and 'spec_visible' not in kwargs:
            new_spec_visible = [{name: True for name in self.spec_names[:self.n_spec]} for _ in range(len(self.exps_data))]

            # update spec_visible and keep the old parameters
            for new_dict, old_dict in zip(new_spec_visible, self.spec_visible):
                for key, value in old_dict.items():
                    if key in new_dict:
                        new_dict[key] = value

            self.spec_visible = new_spec_visible

        if 'n_spec' in kwargs:
            if self.exp_dep_params is not None and self.exp_indep_params is not None:
                all_params = self.get_all_param_names().keys()
                exp_dep_pars_old = self.exp_dep_params

                self.exp_dep_params = self.default_exp_dep_params()
                for old_par in exp_dep_pars_old:
                    if old_par in all_params:
                        if old_par not in self.exp_dep_params:
                            self.exp_dep_params.append(old_par)
                for old_par in self.exp_indep_params:
                    if old_par in all_params:
                        if old_par in self.exp_dep_params:
                            self.exp_dep_params.remove(old_par)

            else:
                self.exp_dep_params = self.default_exp_dep_params()

        if len(kwargs) > 0:
            self._update_params()

    def _update_params(self):
        """Update the parameters based on new model options."""

        old_params = deepcopy(self.params)

        self.init_params()

        if self.params is not None and old_params is not None:
            for key, par in old_params.items():
                if key in self.params:
                    new_par = self.params[key]
                    new_par.value = par.value
                    new_par.vary = par.vary
                    new_par.min = par.min
                    new_par.max = par.max
                    new_par.stderr = par.stderr
                    if hasattr(par, 'enabled'):  # enabled option is added for GUI text fields
                        new_par.enabled = par.enabled

        del old_params

    def init_params(self):
        """Initialized the parameters for current model options"""
        self.params = Parameters()

        if self.exps_data is None or not np.iterable(self.exps_data):
            raise ValueError('Experimental data that are not iterable')

    def simulate(self):
        """Simulates the data and returns the tuple of simulated traces and residuals as lists filled with ndarrays"""
        pass

    def residuals(self):
        """Efficient calculation of only residuals, can be optimized for varpro fitting."""
        pass

    def get_all_param_names(self):
        """returns dictionary of available parameters with keys as names and values as explanations"""
        return dict()

    # def get_available_params_names(self):
    #     return set()

    def is_rate_par(self, par):
        return par.startswith('k_')

    def is_amp_par(self, par):
        return par in self.spec_names

    def is_j_par(self, par):
        return par.startswith('_')

    def is_intercept(self, par):
        return par == 'intercept'

    def format_exp_par(self, par, i):
        return f'{par}_e{i}'

    def default_exp_dep_params(self):
        """Returns the set of default experiment-dependent parameters."""
        pars = self.spec_names[:self.n_spec]
        pars += ['intercept']
        return pars

    def get_ordered_values(self, type: str, exp_num=0):
        return [self.params[param].value for param in self.param_names_dict[exp_num][type]]

    def get_model_indep_params_list(self):
        return [self.params[name] for name in sorted(self.exp_indep_params)]

    def get_model_dep_params_list(self):
        pars_list = []

        for i in range(len(self.exps_data)):
            pars_list.append([self.params[self.format_exp_par(name, i)] for name in sorted(self.exp_dep_params)])

        return pars_list

    def fix_all_exp_dep_params(self, vary: bool, par_name: str):
        if par_name in self.exp_indep_params:
            return

        all_params = map(lambda d: d['all'], self.param_names_dict)

        exp_entry = list(filter(lambda lst: par_name in lst, all_params))[0]

        index = exp_entry.index(par_name)
        for exp_pars in map(lambda d: d['all'], self.param_names_dict):
            parameter = self.params[exp_pars[index]]
            parameter.vary = vary

    def set_all_spec_visible(self, visible: bool, species: str):
        if species not in self.spec_visible[0]:
            return

        for vis_exps in self.spec_visible:
            vis_exps[species] = visible

    def model_options(self):
        """Returns list of dictionaries of all additional options for a given model that
        will be added to fitwidget"""
        # return [dict(type=bool, name='option_name', value=True)]
        return [dict(type=bool, name='varpro', value=True,
                     description='Calculate amplitudes from data by OLS')]


class SeqParModel(Model):

    name = 'Sequential/Parallel Model (1st order)'

    def __init__(self, exps_data: list, ranges: [list, tuple] = None, varpro: bool = True, n_spec=2,
                 exp_dep_params: set = None, spec_visible: dict = None, sequential: bool = True,
                 fit_intercept_varpro=True, weight_func=lambda res, y: res):
        spec_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # use alphabet
        super(SeqParModel, self).__init__(exps_data, ranges=ranges, varpro=varpro, exp_dep_params=exp_dep_params,
                                          n_spec=n_spec, spec_names=spec_names, spec_visible=spec_visible,
                                          weight_func=weight_func)

        self.sequential = sequential
        self.fit_intercept_varpro = fit_intercept_varpro

    def model_options(self):
        opts = super(SeqParModel, self).model_options()
        sequential_opt = dict(type=bool, name='sequential', value=True, description='Use sequential model')
        fit_intercept_varpro_opt = dict(type=bool, name='fit_intercept_varpro', value=True,
                                        description='Calculate intercept from data by OLS')

        return opts + [fit_intercept_varpro_opt, sequential_opt]

    def update_model_options(self, **kwargs):
        super(SeqParModel, self).update_model_options(**kwargs)

        # handle the individual model option changes / override the default implementation
        # also handles visible changes
        for exp_params, visible in zip(self.param_names_dict, self.spec_visible):
            pars_j = [self.params[name] for name in exp_params['j']]
            for i in range(len(pars_j)):
                pars_j[i].vary = False
                pars_j[i].value = 1 if i == 0 or not self.sequential else 0

            for amp, vis in zip((self.params[name] for name in exp_params['amps']), visible.values()):
                cond = not self.varpro and vis and amp.name not in self.exp_indep_params
                amp.vary = cond
                amp.enabled = cond
                if not vis:
                    amp.value = 0

            self.params[exp_params['intercept']].vary = not self.fit_intercept_varpro
            self.params[exp_params['intercept']].enabled = not self.fit_intercept_varpro

        if 'intercept' in self.exp_indep_params:
            self.params[self.param_names_dict[0]['intercept']].enabled = True

        for par_name in self.exp_indep_params:
            if self.is_amp_par(par_name):
                p = self.params[par_name]
                p.vary = True
                p.enabled = True

    def get_all_param_names(self):
        pars = {}
        pars.update({f'_{name}_0': f'Initial concentration of {name}' for name in self.spec_names[:self.n_spec]})
        pars.update({name: f'Amplitude of {name}' for name in self.spec_names[:self.n_spec]})  # amplitudes
        pars.update({f'k_{i+1}': f'Rate constant k_{i+1}' for i in range(self.n_spec)})  # rate constants
        pars.update({'intercept': 'Intercept'})  # intercept
        return pars

    def init_params(self):
        """Initialize the parameters for current model options"""
        super(SeqParModel, self).init_params()

        av_params = self.get_all_param_names().keys()
        # self.exp_dep_params = self.default_exp_dep_params() if self.exp_dep_params is None else self.exp_dep_params
        self.exp_indep_params = []
        for par in av_params:  # keep the order
            if par not in self.exp_dep_params:
                self.exp_indep_params.append(par)

        rates_dict = {f'k_{i+1}': i for i in range(self.n_spec)}
        amps_dict = {name: i for i, name in enumerate(self.spec_names[:self.n_spec])}
        j_dict = {f'_{name}_0': i for i, name in enumerate(self.spec_names[:self.n_spec])}

        def add_params(param_set: list, dict_params: dict,
                       species_visible: [dict] = None, par_format=lambda name: name):
            has_been = False
            for par in param_set:
                f_par_name = par_format(par)

                vary = True
                value = 1
                min = -np.inf
                if self.is_j_par(par):
                    dict_params['j'][j_dict[par]] = f_par_name
                    vary = False
                    if self.sequential:
                        value = 0 if has_been else 1
                        has_been = True

                elif self.is_rate_par(par):
                    dict_params['rates'][rates_dict[par]] = f_par_name
                    min = 0

                elif self.is_amp_par(par):
                    dict_params['amps'][amps_dict[par]] = f_par_name
                    vary = not self.varpro

                    if species_visible is not None and par in species_visible:
                        value = 1 if species_visible[par] else 0
                    else:
                        value = 1
                elif self.is_intercept(par):
                    dict_params['intercept'] = f_par_name
                    vary = self.fit_intercept_varpro and not self.varpro
                    value = 0
                else:
                    value = 0

                dict_params['all'].append(f_par_name)
                self.params.add(f_par_name, min=min, max=np.inf, value=value, vary=vary)
                # add an enabled attribute for each parameter
                self.params[f_par_name].enabled = True

        params_indep = dict(all=[], rates=[''] * len(rates_dict),
                            j=[''] * len(j_dict),
                            amps=[''] * len(amps_dict), intercept='')

        n = len(self.exps_data)

        # add experiment independent parameters
        add_params(self.exp_indep_params, params_indep)

        self.param_names_dict = [deepcopy(params_indep) for _ in range(n)]
        # add experiment dependent parameters
        for i in range(n):
            add_params(self.exp_dep_params, self.param_names_dict[i], species_visible=self.spec_visible[i],
                       par_format=lambda name: self.format_exp_par(name, i))

    def _get_traces(self, t, ks, j):

        n = self.n_spec

        assert n == ks.shape[0]

        if self.sequential:
            # build the transfer matrix for sequential model
            K = np.zeros((n, n))
            for i in range(n):
                K[i, i] = -ks[i]
                if i < n - 1:
                    K[i + 1, i] = ks[i]
            # calculate and return the target model simulation
            return target_1st_order(t, K, j)

        else:  # parallel model
            return j[None, :] * np.heaviside(t[:, None], 1) * np.exp(-t[:, None] * ks[None, :])

    def simulate(self, params=None):
        """Simulates the data and returns the list of simulated traces as ndarrays"""

        if params is not None:
            self.params = params

        if self.exps_data is None or not np.iterable(self.exps_data):
            raise ValueError('No experimental data or data are not iterable')

        x_vals = []
        fits = []
        residuals = []

        lstsq_intercept = self.fit_intercept_varpro and 'intercept' in self.exp_dep_params

        exp_indep_amps = [par for par in self.exp_indep_params if self.is_amp_par(par)]

        for data, x_range, par_names, visible in zip(self.exps_data, self.ranges, self.param_names_dict, self.spec_visible):
            x, y = get_xy(data, x0=x_range[0], x1=x_range[1])
            _y = y.copy()  # copy view of y, it may change, otherwise, original data would be changed
            x_vals.append(x)

            j = np.asarray([self.params[p].value for p in par_names['j']])
            ks = np.asarray([self.params[p].value for p in par_names['rates']])

            traces = self._get_traces(x, ks, j)  # simulate

            if self.varpro:

                amps_params = [self.params[p] for p in par_names['amps']]

                # exp indep traces
                exp_dep_select = []
                exp_indep_select = []
                for key, visible in visible.items():
                    is_independent = key in exp_indep_amps
                    exp_indep_select.append(is_independent and visible)
                    exp_dep_select.append(not is_independent and visible)

                A = traces[:, exp_dep_select]  # select only visible species
                # add intercept as constant function

                fit = 0
                if lstsq_intercept:
                    A = np.hstack((A, np.ones_like(x)[:, None]))
                else:
                    fit = self.params[par_names['intercept']].value
                    _y -= fit

                if any(exp_indep_select):
                    _coefs = np.asarray([p.value for p, indep in zip(amps_params, exp_indep_select) if indep])
                    exp_indep_traces = traces[:, exp_indep_select].dot(_coefs)  # add calculated traces
                    fit += exp_indep_traces
                    _y -= exp_indep_traces

                # solve the least squares problem, find the amplitudes of visible compartments based on data
                coefs = OLS_ridge(A, _y, 0)  # A @ amps = y - A_fixed @ amps_fixes - intercept

                fit += A.dot(coefs)  # calculate the fit and add it

                # update amplitudes and intercept
                if lstsq_intercept:
                    *coefs, intercept = list(coefs)
                    self.params[par_names['intercept']].value = intercept

                amp_names = [amp for amp, selected in zip(amps_params, exp_dep_select) if selected]
                for par, coef in zip(amp_names, coefs):
                    par.value = coef

            else:
                amps = np.asarray([self.params[p].value for p in par_names['amps']])
                fit = traces.dot(amps) + self.params[par_names['intercept']].value  # weight the simulated traces with amplitudes and calculate the fit

            res = self.weight_func(fit - y, y)  # residual, use original data
            fits.append(fit)
            residuals.append(res)

        return x_vals, fits, residuals

    def residuals(self, params=None):
        # TODO--> optimize
        _, _, residuals = self.simulate(params)

        # stack all residuals and return
        return np.hstack(residuals)


class Photosensitization(Model):

    name = 'Photosensitization (PS→, PS+Q→T, T→) [Q]_0 \u226b [PS]_0'

    def __init__(self, exps_data: list, fit_intercept_varpro=True, **kwargs):
        spec_names = ['PS', 'T']
        spec_visible = [{'PS': True, 'T': True} for _ in range(len(exps_data))]
        kwargs.update(n_spec=2, spec_names=spec_names, spec_visible=spec_visible)
        super(Photosensitization, self).__init__(exps_data, **kwargs)

        self.fit_intercept_varpro = fit_intercept_varpro

    def model_options(self):
        opts = super(Photosensitization, self).model_options()
        fit_intercept_varpro_opt = dict(type=bool, name='fit_intercept_varpro', value=True,
                                        description='Calculate intercept from data by OLS')

        return opts + [fit_intercept_varpro_opt]

    def update_model_options(self, **kwargs):
        # handle the individual model option changes / override the default implementation
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f'Argument {key} is not valid.')
            self.__setattr__(key, value)

        if len(kwargs) > 0:
            self._update_params()

        # also handles visible changes
        for exp_params, visible in zip(self.param_names_dict, self.spec_visible):
            for amp, vis in zip((self.params[name] for name in exp_params['amps']), visible.values()):
                amp.vary = not self.varpro and vis
                amp.enabled = not self.varpro and vis
                if not vis:
                    amp.value = 0

            self.params[exp_params['intercept']].vary = not self.fit_intercept_varpro
            self.params[exp_params['intercept']].enabled = not self.fit_intercept_varpro

        if 'intercept' in self.exp_indep_params:
            self.params[self.param_names_dict[0]['intercept']].enabled = True

    def get_all_param_names(self):
        pars = {'_Q_0': 'Concentration of a quencher'}
        pars.update({f'_{name}_0': f'Initial concentration of {name}' for name in self.spec_names})
        pars.update({name: f'Amplitude of {name}' for name in self.spec_names})  # amplitudes
        pars.update({'k_PS': "Decay rate constant of a PS without a quencher"})
        pars.update({'k_q': "Quenching rate constant"})
        pars.update({'k_T': "Decay rate constant of a sensitized triplet T"})
        pars.update({'intercept': 'Intercept'})  # intercept
        return pars

    def default_exp_dep_params(self):
        pars = ['intercept']  # intercepts
        pars += ['_Q_0']  # initial concentration of a quencher
        pars += self.spec_names  # amplitudes
        return pars

    def get_current_species_names(self):
        return self.spec_names

    def init_params(self):
        """Initialize the parameters for current model options"""
        super(Photosensitization, self).init_params()

        if self.exps_data is None or not np.iterable(self.exps_data):
            raise ValueError('Experimental data that are not iterable')

        av_params = set(self.get_all_param_names().keys())
        self.exp_dep_params = self.default_exp_dep_params() if self.exp_dep_params is None else self.exp_dep_params
        self.exp_indep_params = av_params - self.exp_dep_params

        def add_params(param_set: set, dict_params: dict,
                       species_visible: [dict] = None, par_format=lambda name: name):
            for par in param_set:
                f_par_name = par_format(par)

                vary = True
                value = 1
                min = -np.inf
                if self.is_j_par(par):
                    dict_params['j'].append(f_par_name)
                    vary = False
                    if par == '_PS_0' or par == '_Q_0':
                        value = 1
                    else:
                        value = 0
                elif self.is_rate_par(par):
                    dict_params['rates'].append(f_par_name)
                    min = 0
                elif self.is_amp_par(par):
                    dict_params['amps'].append(f_par_name)
                    vary = not self.varpro
                elif self.is_intercept(par):
                    dict_params['intercept'] = f_par_name
                    vary = self.fit_intercept_varpro and not self.varpro
                    value = 0
                else:
                    value = 0

                dict_params['all'].append(f_par_name)
                self.params.add(f_par_name, min=min, max=np.inf, value=value, vary=vary)
                # add an enabled attribute for each parameter
                self.params[f_par_name].enabled = True

        params_indep = dict(all=[], rates=[], j=[], amps=[], intercept='')

        n = len(self.exps_data)

        # add experiment independent parameters
        add_params(self.exp_indep_params, params_indep)

        self.param_names_dict = [deepcopy(params_indep) for _ in range(n)]
        # add experiment dependent parameters
        for i in range(n):
            add_params(self.exp_dep_params, self.param_names_dict[i],
                       species_visible=self.spec_visible[i],
                       par_format=lambda name: self.format_exp_par(name, i))

            self.param_names_dict[i]['all'].sort()
            self.param_names_dict[i]['j'].sort()
            self.param_names_dict[i]['rates'].sort()
            self.param_names_dict[i]['amps'].sort()

    def _get_traces(self, t, ks, j):

        assert ks.shape == j.shape

        # diff equations:
        #   dPS/dt = -k_PS * [PS] - k_q * [Q]*[PS]
        #   dT/dt = + k_q * [Q]*[PS] - k_T * [T]
        #
        #   solution for [Q]_0 >> [PS]_0
        #  c(PS) = [PS]_0 * exp(-(k_PS + k_q * [Q]) * t)
        #  c(T) = ... double exponential

        C = np.empty((t.shape[0], 2), dtype=np.float64)

        j_PS, j_Q, j_T = j
        k_PS, k_T, k_q = ks

        k_form = k_q * j_Q  # formation of T rate constant

        C[:, 0] = j_PS * np.exp(-t * (k_PS + k_form))
        C[:, 1] = j_T + (k_form / (k_T - k_form)) * (np.exp(-k_form * t) - np.exp(-k_T * t))

        return np.nan_to_num(np.heaviside(t[:, None], 1) * C)

    def simulate(self, params=None):
        """Simulates the data and returns the list of simulated traces as ndarrays"""

        if params is not None:
            self.params = params

        if self.exps_data is None or not np.iterable(self.exps_data):
            raise ValueError('No experimental data or data are not iterable')

        x_vals = []
        fits = []
        residuals = []

        lstsq_intercept = self.fit_intercept_varpro and 'intercept' in self.exp_dep_params

        for data, x_range, par_names, visible in zip(self.exps_data, self.ranges, self.param_names_dict, self.spec_visible):
            x, y = get_xy(data, x0=x_range[0], x1=x_range[1])
            x_vals.append(x)

            j = np.asarray([self.params[p].value for p in par_names['j']])
            ks = np.asarray([self.params[p].value for p in par_names['rates']])

            traces = self._get_traces(x, ks, j)  # simulate

            if self.varpro:
                A = traces[:, list(visible.values())]  # select only visible species
                # add intercept as constant function
                if lstsq_intercept:
                    A = np.hstack((A, np.ones_like(x)[:, None]))

                # solve the least squares problem, find the amplitudes of visible compartments based on data
                coefs = OLS_ridge(A, y, 0)

                fit = A.dot(coefs)  # calculate the fit

                # update amplitudes and intercept
                if lstsq_intercept:
                    *coefs, intercept = list(coefs)
                    self.params[par_names['intercept']].value = intercept
                else:
                    fit += self.params[par_names['intercept']].value

                amp_names = [amp_name for amp_name, visible in zip(par_names['amps'], visible.values()) if visible]
                for name, value in zip(amp_names, coefs):
                    self.params[name].value = value

            else:
                amps = np.asarray([self.params[p].value for p in par_names['amps']])
                fit = traces.dot(amps) + self.params[par_names['intercept']].value  # weight the simulated traces with amplitudes and calculate the fit

            res = self.weight_func(fit - y, y)  # residual
            fits.append(fit)
            residuals.append(res)

        return x_vals, fits, residuals

    def residuals(self, params=None):
        # TODO--> optimize
        _, _, residuals = self.simulate(params)

        # stack all residuals and return
        return np.hstack(residuals)


class GeneralFitModel(Model):

    name = 'Sequential/Parallel Model (1st order)'

    def __init__(self, exps_data: list, fit_intercept_varpro=True, show_backward_rates=False, **kwargs):
        super(GeneralFitModel, self).__init__(exps_data, **kwargs)

        self.general_model = GeneralModel()

        self.fit_intercept_varpro = fit_intercept_varpro
        self.show_backward_rates = show_backward_rates

    def model_options(self):
        opts = super(GeneralFitModel, self).model_options()
        fit_intercept_varpro_opt = dict(type=bool, name='fit_intercept_varpro', value=True,
                                        description='Calculate intercept from data by OLS')

        return opts + [fit_intercept_varpro_opt]

    def update_model_options(self, **kwargs):
        super(GeneralFitModel, self).update_model_options(**kwargs)

        # handle the individual model option changes / override the default implementation
        # also handles visible changes
        for exp_params, visible in zip(self.param_names_dict, self.spec_visible):
            for amp, vis in zip((self.params[name] for name in exp_params['amps']), visible.values()):
                cond = not self.varpro and vis and amp.name not in self.exp_indep_params
                amp.vary = cond
                amp.enabled = cond
                if not vis:
                    amp.value = 0

            self.params[exp_params['intercept']].vary = not self.fit_intercept_varpro
            self.params[exp_params['intercept']].enabled = not self.fit_intercept_varpro

        if 'intercept' in self.exp_indep_params:
            self.params[self.param_names_dict[0]['intercept']].enabled = True

        for par_name in self.exp_indep_params:
            if self.is_amp_par(par_name):
                p = self.params[par_name]
                p.vary = True
                p.enabled = True

    def get_all_param_names(self):
        pars = {}

        comps = self.general_model.get_compartments()

        # initial conditions
        pars.update({f'_{name}_0': f'Initial concentration of {name}' for name in comps})
        pars.update({name: f' Amplitude of {name}' for name in comps})  # amplitudes
        pars.update({rate: f'Rate constant {rate}' for rate in
                     self.general_model.get_rates(self.show_backward_rates, False)})  # rate constants
        pars.update({'intercept': 'Intercept'})  # intercept
        return pars

    def get_current_species_names(self):
        return self.general_model.get_compartments()

    def default_exp_dep_params(self):
        """Returns the set of default experiment-dependent parameters."""
        pars = self.general_model.get_compartments()
        pars += ['intercept']  # intercept
        return pars

    def get_model_indep_params_list(self):
        return [self.params[name] for name in self.exp_indep_params]

    def get_model_dep_params_list(self):
        pars_list = []

        for i in range(len(self.exps_data)):
            pars_list.append([self.params[self.format_exp_par(name, i)] for name in self.exp_dep_params])

        return pars_list

    def load_from_file(self, fpath: str):
        self.general_model = GeneralModel.load(fpath)
        self.build()

    def load_from_scheme(self, scheme: str):
        self.general_model = GeneralModel.from_text(scheme)
        self.build()

    def build(self):
        self.general_model.build_func()
        comps = self.general_model.get_compartments()
        self.update_model_options(n_spec=len(comps), spec_names=comps)

    def __getattr__(self, item):
        return self.general_model.__getattribute__(item)

    def init_params(self):
        """Initialize the parameters for current model options"""
        super(GeneralFitModel, self).init_params()

        av_params = self.get_all_param_names().keys()
        self.exp_dep_params = self.default_exp_dep_params() if self.exp_dep_params is None else self.exp_dep_params
        self.exp_indep_params = []
        for par in av_params:  # keep the order
            if par not in self.exp_dep_params:
                self.exp_indep_params.append(par)

        rates_vals = self.general_model.get_rates(self.show_backward_rates, True)
        rates_dict = {rate[0]: i for i, rate in enumerate(rates_vals)}
        amps_dict = {name: i for i, name in enumerate(self.spec_names)}
        j_dict = {f'_{name}_0': i for i, name in enumerate(self.spec_names)}

        def add_params(param_list: list, dict_params: dict, par_format=lambda name: name):
            for par in param_list:
                f_par_name = par_format(par)

                vary = True
                value = 1
                min = -np.inf
                if self.is_j_par(par):
                    dict_params['j'][j_dict[par]] = f_par_name  # place at correct position
                    try:
                        # get species name from j format = remove initial _ sign and last 2 characters
                        value = self.general_model.initial_conditions[par[1:-2]]
                    except KeyError:
                        pass
                    vary = False
                elif self.is_rate_par(par):
                    dict_params['rates'][rates_dict[par]] = f_par_name

                    f = list(filter(lambda r: par == r[0], rates_vals))  # get rates from general model
                    if len(f) > 0:
                        value = f[0][1]
                    min = 0
                elif self.is_amp_par(par):
                    dict_params['amps'][amps_dict[par]] = f_par_name
                    vary = not self.varpro
                elif self.is_intercept(par):
                    dict_params['intercept'] = f_par_name
                    vary = self.fit_intercept_varpro and not self.varpro
                    value = 0
                else:
                    value = 0

                self.params.add(f_par_name, min=min, max=np.inf, value=value, vary=vary)
                dict_params['all'].append(f_par_name)
                # add an enabled attribute for each parameter
                self.params[f_par_name].enabled = True

        params_indep = dict(all=[], rates=[''] * len(rates_dict),
                            j=[''] * len(j_dict),
                            amps=[''] * len(amps_dict), intercept='')

        n = len(self.exps_data)

        # add experiment independent parameters
        add_params(self.exp_indep_params, params_indep)

        self.param_names_dict = [deepcopy(params_indep) for _ in range(n)]
        # add experiment dependent parameters
        for i in range(n):
            add_params(self.exp_dep_params, self.param_names_dict[i],
                       par_format=lambda name: self.format_exp_par(name, i))

    def _get_traces(self, x_data, rates, j):
        if x_data[0] > 0:  # initial conditions are valid for time=0
            n = 100  # prepend x values with 100 points if not starting with zero time
            x_prepended = np.concatenate((np.linspace(0, x_data[0], n, endpoint=False), x_data))
            return odeint(self.general_model.func, j, x_prepended, args=(rates,))[n:, :]

        elif x_data[0] < 0:
            x_pos = x_data[x_data >= 0]  # find x >= 0
            sol = np.zeros((x_data.shape[0], j.shape[0]), dtype=np.float64)
            if x_pos.shape[0] > 1:  # simulate only for at least 2 positive values
                sol[(x_data < 0).sum():, :] = self._get_traces(x_pos, rates, j)  # use recursion here

            return sol

        # for x_data[0] == 0
        return odeint(self.general_model.func, j, x_data, args=(rates,))

    def get_rate_values(self, exp_num=0):
        ks = np.asarray([self.params[p].value for p in self.param_names_dict[exp_num]['rates']])

        # transform ks to 2d array, needed for general model, first column are forward rates, the second backward rates
        if self.show_backward_rates:
            rates = np.empty((ks.shape[0] // 2, 2))
            rates[:, 0] = ks[::2]
            rates[:, 1] = ks[1::2]
        else:
            rates = np.vstack((ks, np.zeros(ks.shape[0]))).T

        return rates

    def simulate(self, params=None):
        """Simulates the data and returns the list of simulated traces as ndarrays"""

        if params is not None:
            self.params = params

        if self.exps_data is None or not np.iterable(self.exps_data):
            raise ValueError('No experimental data or data are not iterable')

        x_vals = []
        fits = []
        residuals = []

        lstsq_intercept = self.fit_intercept_varpro and 'intercept' in self.exp_dep_params

        exp_indep_amps = [par for par in self.exp_indep_params if self.is_amp_par(par)]

        i = 0
        for data, x_range, par_names, visible in zip(self.exps_data, self.ranges, self.param_names_dict, self.spec_visible):
            x, y = get_xy(data, x0=x_range[0], x1=x_range[1])
            _y = y.copy()  # copy view of y, it may change, otherwise, original data would be changed
            x_vals.append(x)

            j = np.asarray([self.params[p].value for p in par_names['j']])
            rates = self.get_rate_values(i)

            traces = self._get_traces(x, rates, j)  # simulate

            if self.varpro:

                amps_params = [self.params[p] for p in par_names['amps']]

                # exp indep traces
                exp_dep_select = []
                exp_indep_select = []
                for key, visible in visible.items():
                    is_independent = key in exp_indep_amps
                    exp_indep_select.append(is_independent and visible)
                    exp_dep_select.append(not is_independent and visible)

                A = traces[:, exp_dep_select]  # select only visible species
                # add intercept as constant function

                fit = 0
                if lstsq_intercept:
                    A = np.hstack((A, np.ones_like(x)[:, None]))
                else:
                    fit = self.params[par_names['intercept']].value
                    _y -= fit

                if any(exp_indep_select):
                    _coefs = np.asarray([p.value for p, indep in zip(amps_params, exp_indep_select) if indep])
                    exp_indep_traces = traces[:, exp_indep_select].dot(_coefs)  # add calculated traces
                    fit += exp_indep_traces
                    _y -= exp_indep_traces

                # solve the least squares problem, find the amplitudes of visible compartments based on data
                coefs = OLS_ridge(A, _y, 0)  # A @ amps = y - A_fixed @ amps_fixes - intercept

                fit += A.dot(coefs)  # calculate the fit and add it

                # update amplitudes and intercept
                if lstsq_intercept:
                    *coefs, intercept = list(coefs)
                    self.params[par_names['intercept']].value = intercept

                amp_names = [amp for amp, selected in zip(amps_params, exp_dep_select) if selected]
                for par, coef in zip(amp_names, coefs):
                    par.value = coef

            else:
                amps = np.asarray([self.params[p].value for p in par_names['amps']])
                fit = traces.dot(amps) + self.params[par_names['intercept']].value  # weight the simulated traces with amplitudes and calculate the fit

            res = self.weight_func(fit - y, y)  # residual, use original data
            fits.append(fit)
            residuals.append(res)
            i += 1

        return x_vals, fits, residuals

    def residuals(self, params=None):
        # TODO--> optimize
        _, _, residuals = self.simulate(params)

        # stack all residuals and return
        return np.hstack(residuals)



#
#
# # abstract class for specific model
# class _Model(object):
#     name = '[model name]'
#
#     def __init__(self):
#         self.params = Parameters()
#         self.init_params()
#
#     def init_params(self):
#         pass
#
#     def initialize_values(self, x_data, y_data):
#         pass
#
#     @staticmethod
#     def func(x, *params):
#         pass
#
#     def wrapper_func(self, x, params):
#         """Gets values from lmfit parameters and passes them into function func."""
#         par_tup = (p.value for p in params.values())
#         return self.func(x, *par_tup)
#
#     def get_func_string(self) -> str:
#         """Returns the implementation of the specific model func as a string."""
#         text = inspect.getsource(self.func)
#
#         it = iter(text.splitlines(keepends=True))
#         # skip first two lines
#         it.__next__()
#         it.__next__()
#
#         buffer = ""
#         # 'remove' two tabs = 8 spaces from each line
#         for line in it:
#             buffer += line[8:]
#
#         return buffer
#
#     def par_count(self) -> int:
#         """Returns the number of parameters for this model."""
#         return self.params.__len__()
#
#
# class AB1stModel(_Model):
#     """This is the A→B model, 1st order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k[\mathrm{A}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k[\mathrm{A}]`
#
#     initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = 0`
#     """
#
#     name = 'A→B (1st order)'
#
#     def init_params(self):
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#
#     @staticmethod
#     def func(x, A, B, k, y0):
#         cA = np.exp(-k * x)
#         return A * cA + B * (1 - cA) + y0
#
#
# class ABVarOrderModel(_Model):
#     """This is the A→B model, variable order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k[\mathrm{A}]^n`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k[\mathrm{A}]^n`
#
#     initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = 0`
#     """
#
#     name = 'A→B (var order)'
#
#     def init_params(self):
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('n', value=1, min=0, max=100, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#
#     @staticmethod
#     def func(x, c0, A, B, k, n, y0):
#         if n == 1:  # first order
#             cA = c0 * np.exp(-k * x)
#         else:  # n-th order, this wont work for negative x values, c(t) = 1-n root of (c^(1-n) + k*(n-1)*t)
#             expr_in_root = np.power(float(c0), 1 - n) + k * (n - 1) * x
#             expr_in_root = expr_in_root.clip(min=0)  # set to 0 all the negative values
#             cA = np.power(expr_in_root, 1.0 / (1 - n))
#
#         return A * cA + B * (1 - cA) + y0
#
#
# class AB_mixed12Model(_Model):
#     """This is the A→B model, mixed 1st and 2nd order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}] - k_2[\mathrm{A}]^2`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] + k_2[\mathrm{A}]^2`
#
#     initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = 0`
#     """
#
#     name = 'A→B (Mixed 1st and 2nd order)'
#
#     def init_params(self):
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#
#     @staticmethod
#     def func(x, c0, A, B, k1, k2, y0):
#         cA = c0 * k1 / ((c0 * k2 + k1) * np.exp(k1 * x) - c0 * k2)
#         return A * cA + B * (1 - cA) + y0
#
#
# class ABC1stModel(_Model):
#     """This is the A→B→C model, 1st order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] - k_2[\mathrm{B}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]`
#
#     initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = 0`
#     """
#
#     name = 'A→B→C (1st order)'
#
#     def init_params(self):
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('C', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#         self.params['B'].value = y_data[0]
#
#     @staticmethod
#     def func(x, A, B, C, k1, k2, y0):
#         cA = np.exp(-k1 * x)
#
#         # if k1 == k2, the integrated law fow cB has to be changed
#         if np.abs(k1 - k2) < 1e-10:
#             cB = k1 * x * np.exp(-k1 * x)
#         else:  # we would get zero division error for k1 = k2
#             cB = (k1 / (k2 - k1)) * (np.exp(-k1 * x) - np.exp(-k2 * x))
#
#         cC = 1 - cA - cB
#
#         return A * cA + B * cB + C * cC + y0
#
#
# class ABCvarOrderModel(_Model):
#     """This is the A→B→C model, variable order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]^{n_1}`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}]^{n_1} - k_2[\mathrm{B}]^{n_2}`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]^{n_2}`
#
#     initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = 0`
#
#     Note, that :math:`n_2` must be :math:`\ge 1`, however, :math:`n_1 \ge 0`.
#     """
#
#     name = 'A→B→C (var order)'
#
#     def init_params(self):
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('C', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('n1', value=1, min=0, max=100, vary=False)
#         # numerical integration won't work for n < 1, therefore, min=1
#         self.params.add('n2', value=1, min=1, max=100, vary=False)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#         self.params['B'].value = y_data[0]
#
#     @staticmethod
#     def func(x, c0, A, B, C, k1, k2, n1, n2, y0):
#         def cA(c0, k, n, x):
#             if n == 1:
#                 return c0 * np.exp(-k * x)
#             else:
#                 expr_in_root = c0 ** (1 - n) + k * (n - 1) * x
#                 # we have to set all negative values to 0 (this is important for n < 1),
#                 # because the root of negative value would cause error
#                 if type(expr_in_root) is float:
#                     expr_in_root = 0 if expr_in_root < 0 else expr_in_root
#                 else:
#                     expr_in_root = expr_in_root.clip(min=0)
#                 return np.power(expr_in_root, 1.0 / (1 - n))
#
#         # definition of differential equations
#         def dB_dt(cB, x):
#             return k1 * cA(c0, k1, n1, x) ** n1 - k2 * cB ** n2
#
#         # solve for cB, cA is integrated law, initial conditions, cB(t=0)=0
#         # but initial point for odeint must be defined for x[0]
#         # evolve from 0 to x[0] and then take the last point as the initial condition
#         x0 = np.linspace(0, x[0], num=100)
#         init_B = odeint(dB_dt, 0, x0).flatten()[-1]
#         result = odeint(dB_dt, init_B, x)
#
#         cA = cA(c0, k1, n1, x)
#         cB = result.flatten()
#         cC = c0 - cA - cB
#
#         return A * cA + B * cB + C * cC + y0
#
#
# class ABCD1stModel(_Model):
#     """This is the A→B→C→D model, 1st order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] - k_2[\mathrm{B}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}] - k_3[\mathrm{C}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{D}]}{\mathrm{d}t} = +k_3[\mathrm{C}]`
#
#     initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = [\mathrm D]_0 = 0`
#     """
#     name = 'A→B→C→D (1st order)'
#
#     def init_params(self):
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('C', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('D', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('k3', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#         self.params['B'].value = y_data[0]
#         self.params['C'].value = y_data[0]
#
#     @staticmethod
#     def func(x, A, B, C, D, k1, k2, k3, y0):
#         def concB(k1, k2, x):
#             # if k1 == k2, the integrated law fow cB has to be changed
#             if np.abs(k1 - k2) < 1e-10:
#                 return k1 * x * np.exp(-k1 * x)
#             else:  # we would get zero division error for k1 = k2
#                 return (k1 / (k2 - k1)) * (np.exp(-k1 * x) - np.exp(-k2 * x))
#
#         def dC_dt(cC, t):
#             cB = concB(k1, k2, t)
#             return k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]
#
#         # numerically integrate, initial condition, cC(t=0) = 0
#         # but initial point for odeint must be defined for x[0]
#         # evolve from 0 to x[0] and then take the last point as the initial condition
#         x0 = np.linspace(0, x[0], num=100)
#         init_C = odeint(dC_dt, 0, x0).flatten()[-1]
#         result = odeint(dC_dt, init_C, x)
#
#         cA = np.exp(-k1 * x)
#         cB = concB(k1, k2, x)
#         cC = result.flatten()
#         cD = 1 - cA - cB - cC
#
#         return A * cA + B * cB + C * cC + D * cD + y0
#
#
# class ABCDvarOrderModel(_Model):
#     """This is the A→B→C→D model, variable order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]^{n_1}`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}]^{n_1} - k_2[\mathrm{B}]^{n_2}`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]^{n_2} - k_3[\mathrm{C}]^{n_3}`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{D}]}{\mathrm{d}t} = +k_3[\mathrm{C}]^{n_3}`
#
#     initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = [\mathrm D]_0 = 0`
#
#     Note, that :math:`n_2,n_3` must be :math:`\ge 1`, however, :math:`n_1 \ge 0`.
#     """
#     name = 'A→B→C→D (var order)'
#
#     def init_params(self):
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('C', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('D', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('k3', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('n1', value=1, min=0, max=100, vary=False)
#         # numerical integration won't work for n < 1, therefore, min=1
#         self.params.add('n2', value=1, min=1, max=100, vary=False)
#         self.params.add('n3', value=1, min=1, max=100, vary=False)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#         self.params['B'].value = y_data[0]
#         self.params['C'].value = y_data[0]
#
#     @staticmethod
#     def func(x, c0, A, B, C, D, k1, k2, k3, n1, n2, n3, y0):
#         def cA(c0, k, n, x):
#             if n == 1:
#                 return c0 * np.exp(-k * x)
#             else:
#                 expr_in_root = c0 ** (1 - n) + k * (n - 1) * x
#                 # we have to set all negative values to 0 (this is important for n < 1),
#                 # because the root of negative value would cause error
#                 if type(expr_in_root) is float:
#                     expr_in_root = 0 if expr_in_root < 0 else expr_in_root
#                 else:
#                     expr_in_root = expr_in_root.clip(min=0)
#                 return np.power(expr_in_root, 1.0 / (1 - n))
#
#         # definition of differential equations
#         def solve(concentrations, x):
#             cB, cC = concentrations
#             dB_dt = k1 * cA(c0, k1, n1, x) ** n1 - k2 * cB ** n2
#             dC_dt = k2 * cB ** n2 - k3 * cC ** n3
#             return [dB_dt, dC_dt]
#
#         # solve for cB and cC, cA is integrated law, initial conditions, cB(t=0)=cC(t=0)=0
#         # but initial point for odeint must be defined for x[0]
#         # evolve from 0 to x[0] and then take the last points as the initial condition
#         x0 = np.linspace(0, x[0], num=100)
#         init_BC = odeint(solve, [0, 0], x0)[-1, :]  # take the last two points in the result matrix
#         result = odeint(solve, init_BC, x)
#
#         cA = cA(c0, k1, n1, x)
#         cB = result[:, 0]
#         cC = result[:, 1]
#         cD = c0 - cA - cB - cC
#
#         return A * cA + B * cB + C * cC + D * cD + y0
#
#
# class ABCD_parModel(_Model):
#     """This is the A→B, C→D parallel model, 1st order kinetics. Differential equations:
#
#     :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = -k_2[\mathrm{C}]`
#
#     :math:`\\frac{\mathrm{d}[\mathrm{D}]}{\mathrm{d}t} = +k_2[\mathrm{C}]`
#
#     initial conditions are same as in A→B model, since it is a 1st order, :math:`c_0` was not added to
#     the parameters.
#     """
#     name = 'A→B, C→D (1st order)'
#
#     def init_params(self):
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('C', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('D', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = y_data[0]
#         self.params['C'].value = y_data[0]
#
#     @staticmethod
#     def func(x, A, B, C, D, k1, k2, y0):
#         cA = np.exp(-k1 * x)
#         cC = np.exp(-k2 * x)
#
#         return A * cA + B * (1 - cA) + C * cC + D * (1 - cC) + y0
#
#
# class LinearModel(_Model):
#     """This is the linear model,
#     :math:`y(x) = ax + y_0`
#     where :math:`a` is a slope and :math:`y_0` intercept.
#     """
#     name = 'Linear'
#
#     def init_params(self):
#         self.params.add('a', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('y0', value=1, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['a'].value = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
#         self.params['y0'].value = y_data[0]
#
#     @staticmethod
#     def func(x, a, y0):
#         return a * x + y0
#
#
# class Photobleaching_Model(_Model):
#     """This is the photobleaching model. Differential equation:
#
#     :math:`\\frac{\mathrm{d}c(t)}{\mathrm{d}t} = -k\\left(1 - 10^{-\\varepsilon c(t)l}\\right)`
#
#     where :math:`k = \\frac{1}{V}I_0\Phi` is a product of incident light intensity and photoreaction quantum yield
#     divided by the volume of the solution.
#
#     initial conditions: :math:`c(0) = c_0 = \\frac{A_0}{\\varepsilon l}`
#
#     Integrated law for evolution of absorbance in time (fitting equation):
#
#     :math:`A(t) = \\frac{1}{\\ln 10}\\ln\\left(  10^{-A_0k^{\\prime}t}  \\left( 10^{A_0} + 10^{A_0k^{\\prime}t} - 1  \\right)\\right)`
#
#     where :math:`A_0` is absorbance at time 0, :math:`k^{\\prime} = \\frac{k}{c_0}`, both :math:`k` and :math:`c_0`
#     are fitting parameters, one of them must be fixed, usually :math:`c_0`
#     """
#     name = 'Photobleaching'
#
#     def init_params(self):
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('A0', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('k', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         self.params['A0'].value = y_data[0]
#
#     @staticmethod
#     def func(x, c0, A0, k, y0):
#         kp = k / c0  # k'
#         return np.log(10 ** (-A0 * kp * x) * (10 ** A0 + 10 ** (A0 * kp * x) - 1)) / np.log(10) + y0
#
#
# class Eyring(_Model):
#     """This is the Eyring model,
#     :math:`k(T) = \\frac{\\kappa k_B T}{h}e^{\\frac{\\Delta S}{R}}e^{-\\frac{\\Delta H}{RT}}`
#     """
#     name = 'Eyring'
#
#     def init_params(self):
#         self.params.add('dH', value=1000, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('dS', value=100, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('kappa', value=1, min=0, max=1, vary=False)
#
#     def initialize_values(self, x_data, y_data):
#         # self.params['a'].value = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
#         # self.params['y0'].value = y_data[0]
#         pass
#
#     @staticmethod
#     def func(x, dH, dS, kappa):
#         kB = 1.38064852 * 1e-23  # boltzmann
#         h = 6.62607004 * 1e-34  # planck
#         R = 8.31446261815324  # universal gas constant
#         # x - thermodynamic temperature
#         return (kappa * kB * x / h) * np.exp((dS * x - dH) / (R * x))
