# from abc import abstractmethod
# from abc import ABC
# import inspect

from lmfit import Parameters
import numpy as np
from scipy.integrate import odeint
from ..spectrum import fi
import scipy
from copy import deepcopy
from scipy.linalg import lstsq
from ..general_model import GeneralModel
import numba as nb
import ray
import psutil

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

posv = scipy.linalg.get_lapack_funcs(('posv'))
# gels = scipy.linalg.get_lapack_funcs(('gels'))


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
    """
    Fast least squares solver for X: AX=B by direct solve method with optional Tikhonov regularization.
    """

    ATA = A.T.dot(A)
    ATB = A.T.dot(B)

    if alpha != 0:
        ATA.flat[::ATA.shape[-1] + 1] += alpha

    # call the LAPACK function, direct solve method
    c, x, info = posv(ATA, ATB, lower=False,
                      overwrite_a=False,
                      overwrite_b=False)

    return x


@nb.vectorize(nopython=True)
def parallel_model(j, t, k):
    """Exponential model with simple implementation of heaviside function.
    j is initial population vector, t is time and k is rate constant"""
    if t >= 0:
        return j * np.exp(-t * k)
    else:
        return 0

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
        self.equal_pars = {}  # hold a exp_dep_param as a key and list tuple of pairs of equal parameters as a value

        if self.spec_visible is None:
            self.spec_visible = [{name: True for name in self.spec_names[:self.n_spec]} for _ in range(len(self.exps_data))]

        self.param_names_dict = {}

        self.data_ids = [ray.put(data) for data in self.exps_data]

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
            self.update_params()

    def update_params(self):
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
        return [self.params[name] for name in self.exp_indep_params]

    def get_model_dep_params_list(self):
        pars_list = []

        # find experiment-dependent parameters form 'all' field
        for i in range(len(self.exps_data)):
            exp_pars = []
            for exp_dep_par in self.exp_dep_params:
                for par in self.param_names_dict[i]['all']:
                    if par.startswith(exp_dep_par):
                        exp_pars.append(self.params[par])

            pars_list.append(exp_pars)

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


class _InterceptVarProModel(Model):

    def __init__(self, *args, fit_intercept_varpro=True, **kwargs):
        super(_InterceptVarProModel, self).__init__(*args, **kwargs)

        self.fit_intercept_varpro = fit_intercept_varpro

    def set_equal_param_each(self, par_name: str = '_Q_0', n: int = 2):
        """Fills the equal_pars dictionary for a given par_name so that each n subsequent
        experiments will share the same parameter. Eg. for 6 different experiment and
        parameter _Q_0 and n=2, the resulting pairs will be
        equal_pars['_Q_0'] = [(0, 1), (2, 3), (4, 5)] which means that there will be 3
        different _Q_0 parameters for these 3 experiment pairs. First two experiments
        will use _Q_0_e01 parameter, another two experiments parameter _Q_0_e23, etc."""

        n_exps = len(self.exps_data)
        if n_exps % n != 0:
            raise ValueError(f"Number of experiments is not divisible by {n}.")

        space = np.arange(0, n_exps).reshape((n_exps // n, n))
        pairs = [tuple(row) for row in space]
        self.equal_pars[par_name] = pairs
        self.update_params()

    def model_options(self):
        opts = super(_InterceptVarProModel, self).model_options()
        fit_intercept_varpro_opt = dict(type=bool, name='fit_intercept_varpro', value=True,
                                        description='Calculate intercept from data by OLS')

        return opts + [fit_intercept_varpro_opt]

    def _get_traces(self, t, ks, j, i):
        """i is the current exp index"""
        raise NotImplementedError()

    def add_params(self, param_set: list = None, dict_params: dict = None, j_dict: dict = None, rates_dict: dict = None,
                   amps_dict: dict = None, species_visible: [dict] = None, par_format=lambda name: name, **kwargs):
        pass

    def get_param_dicts(self):
        rates_dict = {f'k_{i + 1}': i for i in range(self.n_spec)}
        amps_dict = {name: i for i, name in enumerate(self.spec_names[:self.n_spec])}
        j_dict = {f'_{name}_0': i for i, name in enumerate(self.spec_names[:self.n_spec])}

        return dict(rates_dict=rates_dict, amps_dict=amps_dict, j_dict=j_dict)

    def init_params(self):
        """Initialize the parameters for current model options"""
        super(_InterceptVarProModel, self).init_params()

        av_params = self.get_all_param_names().keys()
        # self.exp_dep_params = self.default_exp_dep_params() if self.exp_dep_params is None else self.exp_dep_params
        self.exp_indep_params = []
        for par in av_params:  # keep the order
            if par not in self.exp_dep_params:
                self.exp_indep_params.append(par)

        kwargs = self.get_param_dicts()

        params_indep = dict(all=[], rates=[''] * len(kwargs['rates_dict']),
                            j=[''] * len(kwargs['j_dict']),
                            amps=[''] * len(kwargs['amps_dict']), intercept='')

        n = len(self.exps_data)

        # add experiment independent parameters
        self.add_params(self.exp_indep_params, params_indep, **kwargs)

        self.param_names_dict = [deepcopy(params_indep) for _ in range(n)]

        # setup equal parameter names
        equal_pars_dict = {}
        for par_name, pairs in self.equal_pars.items():
            equal_pars_dict[par_name] = ['' for _ in range(len(self.exps_data))]

            for pair in pairs:
                idx_name = ''.join([str(i) for i in pair])
                for idx in pair:
                    equal_pars_dict[par_name][idx] = idx_name

        # add experiment dependent parameters
        for i in range(n):
            def par_format(name):
                if name in equal_pars_dict:
                    idx = equal_pars_dict[name][i]
                    if idx != '':
                        return self.format_exp_par(name, idx)

                return self.format_exp_par(name, i)

            self.add_params(self.exp_dep_params, self.param_names_dict[i], **kwargs,
                            species_visible=self.spec_visible[i],
                            par_format=par_format)

    def get_rate_values(self, exp_num):
        return np.asarray([self.params[p].value for p in self.param_names_dict[exp_num]['rates']])

    @ray.remote
    def simulate_parallel(self, i, data, x_range, par_names, visible):

        lstsq_intercept = self.fit_intercept_varpro and 'intercept' in self.exp_dep_params
        exp_indep_amps = [par for par in self.exp_indep_params if self.is_amp_par(par)]

        x, y = get_xy(data, x0=x_range[0], x1=x_range[1])
        _y = y.copy()  # copy view of y, it may change, otherwise, original data would be changed

        j = np.asarray([self.params[p].value for p in par_names['j']])
        rates = self.get_rate_values(i)

        traces = self._get_traces(x, rates, j, i)  # simulate

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

            if any(exp_indep_select):  # calculate traces for independent-exp amplitudes and add to fit
                _amps = np.asarray([p.value for p, indep in zip(amps_params, exp_indep_select) if indep])
                exp_indep_traces = traces[:, exp_indep_select].dot(_amps)  # add calculated traces
                fit += exp_indep_traces
                _y -= exp_indep_traces

            # solve the least squares problem, find the amplitudes of visible compartments based on data
            amps = OLS_ridge(A, _y, 0)  # A @ amps = y - A_fixed @ amps_fixes - intercept

            fit += A.dot(amps)  # calculate the fit and add it

            # update amplitudes and intercept
            if lstsq_intercept:
                *amps, intercept = list(amps)
                self.params[par_names['intercept']].value = intercept

            amp_names = [amp for amp, selected in zip(amps_params, exp_dep_select) if selected]
            for par, coef in zip(amp_names, amps):
                par.value = coef

        else:
            amps = np.asarray([self.params[p].value for p in par_names['amps']])
            fit = traces.dot(amps)  # weight the simulated traces with amplitudes and calculate the fit

            if lstsq_intercept:
                intercept = (y - fit).sum() / fit.shape[0]  # calculate intercept by least squares
                fit += intercept
                self.params[par_names['intercept']].value = intercept
            else:
                fit += self.params[par_names['intercept']].value  # just add it to fit

        res = self.weight_func(fit - y, y)  # residual, use original data

        return x, fit, res

    def simulate(self, params=None):

        if params is not None:
            self.params = params

        if self.exps_data is None or not np.iterable(self.exps_data):
            raise ValueError('No experimental data or data are not iterable')

        ids = []

        for i, (x_range, par_names, visible) in enumerate(zip(self.ranges, self.param_names_dict,
                                                            self.spec_visible)):

            ids.append(self.simulate_parallel.remote(self, i, self.data_ids[i], x_range, par_names, visible))

        results = ray.get(ids)

        x_vals = list(map(lambda r: r[0], results))
        fits = list(map(lambda r: r[1], results))
        residuals = list(map(lambda r: r[2], results))

        return x_vals, fits, residuals



    # def simulate(self, params=None):
    #     """Simulates the data and returns the list of simulated traces as ndarrays"""
    #
    #     if params is not None:
    #         self.params = params
    #
    #     if self.exps_data is None or not np.iterable(self.exps_data):
    #         raise ValueError('No experimental data or data are not iterable')
    #
    #     x_vals = []
    #     fits = []
    #     residuals = []
    #
    #     lstsq_intercept = self.fit_intercept_varpro and 'intercept' in self.exp_dep_params
    #
    #     exp_indep_amps = [par for par in self.exp_indep_params if self.is_amp_par(par)]
    #
    #     for i, (data, x_range, par_names, visible) in enumerate(zip(self.exps_data, self.ranges, self.param_names_dict,
    #                                                                 self.spec_visible)):
    #         x, y = get_xy(data, x0=x_range[0], x1=x_range[1])
    #         _y = y.copy()  # copy view of y, it may change, otherwise, original data would be changed
    #         x_vals.append(x)
    #
    #         j = np.asarray([self.params[p].value for p in par_names['j']])
    #         rates = self.get_rate_values(i)
    #
    #         traces = self._get_traces(x, rates, j, i)  # simulate
    #
    #         if self.varpro:
    #
    #             amps_params = [self.params[p] for p in par_names['amps']]
    #
    #             # exp indep traces
    #             exp_dep_select = []
    #             exp_indep_select = []
    #             for key, visible in visible.items():
    #                 is_independent = key in exp_indep_amps
    #                 exp_indep_select.append(is_independent and visible)
    #                 exp_dep_select.append(not is_independent and visible)
    #
    #             A = traces[:, exp_dep_select]  # select only visible species
    #             # add intercept as constant function
    #
    #             fit = 0
    #             if lstsq_intercept:
    #                 A = np.hstack((A, np.ones_like(x)[:, None]))
    #             else:
    #                 fit = self.params[par_names['intercept']].value
    #                 _y -= fit
    #
    #             if any(exp_indep_select):  # calculate traces for independent-exp amplitudes and add to fit
    #                 _amps = np.asarray([p.value for p, indep in zip(amps_params, exp_indep_select) if indep])
    #                 exp_indep_traces = traces[:, exp_indep_select].dot(_amps)  # add calculated traces
    #                 fit += exp_indep_traces
    #                 _y -= exp_indep_traces
    #
    #             # solve the least squares problem, find the amplitudes of visible compartments based on data
    #             amps = OLS_ridge(A, _y, 0)  # A @ amps = y - A_fixed @ amps_fixes - intercept
    #
    #             fit += A.dot(amps)  # calculate the fit and add it
    #
    #             # update amplitudes and intercept
    #             if lstsq_intercept:
    #                 *amps, intercept = list(amps)
    #                 self.params[par_names['intercept']].value = intercept
    #
    #             amp_names = [amp for amp, selected in zip(amps_params, exp_dep_select) if selected]
    #             for par, coef in zip(amp_names, amps):
    #                 par.value = coef
    #
    #         else:
    #             amps = np.asarray([self.params[p].value for p in par_names['amps']])
    #             fit = traces.dot(amps)  # weight the simulated traces with amplitudes and calculate the fit
    #
    #             if lstsq_intercept:
    #                 intercept = (y - fit).sum() / fit.shape[0]  # calculate intercept by least squares
    #                 fit += intercept
    #                 self.params[par_names['intercept']].value = intercept
    #             else:
    #                 fit += self.params[par_names['intercept']].value  # just add it to fit
    #
    #         res = self.weight_func(fit - y, y)  # residual, use original data
    #         fits.append(fit)
    #         residuals.append(res)
    #
    #     return x_vals, fits, residuals

    def residuals(self, params=None):
        _, _, residuals = self.simulate(params)

        # stack all residuals and return
        return np.hstack(residuals)


class SeqParModel(_InterceptVarProModel):

    name = 'Sequential/Parallel Model (1st order)'

    def __init__(self, *args, sequential=True, **kwargs):
        spec_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # use alphabet
        kwargs.update(spec_names=spec_names)
        super(SeqParModel, self).__init__(*args, **kwargs)

        self.sequential = sequential

    def model_options(self):
        opts = super(SeqParModel, self).model_options()
        sequential_opt = dict(type=bool, name='sequential', value=True, description='Use sequential model')

        return opts + [sequential_opt]

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

    def get_param_dicts(self):
        rates_dict = {f'k_{i + 1}': i for i in range(self.n_spec)}
        amps_dict = {name: i for i, name in enumerate(self.spec_names[:self.n_spec])}
        j_dict = {f'_{name}_0': i for i, name in enumerate(self.spec_names[:self.n_spec])}

        return dict(rates_dict=rates_dict, amps_dict=amps_dict, j_dict=j_dict)

    def add_params(self, param_set: list = None, dict_params: dict = None, j_dict: dict = None, rates_dict: dict = None,
                   amps_dict: dict = None, species_visible: [dict] = None, par_format=lambda name: name, **kwargs):
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
            if f_par_name not in self.params:
                self.params.add(f_par_name, min=min, max=np.inf, value=value, vary=vary)
                # add an enabled attribute for each parameter
                self.params[f_par_name].enabled = True

    def _get_traces(self, t, ks, j, i):

        n = self.n_spec

        assert n == ks.shape[0]

        if self.sequential:
            # build the transfer matrix for sequential model
            K = np.zeros((n, n))
            for idx in range(n):
                K[idx, idx] = -ks[idx]
                if idx < n - 1:
                    K[idx + 1, idx] = ks[idx]
            # calculate and return the target model simulation
            return target_1st_order(t, K, j)

        else:  # parallel model
            return parallel_model(j[None, :], t[:, None], ks[None, :])


class _FixedParametersModel(_InterceptVarProModel):

    def update_model_options(self, **kwargs):
        # handle the individual model option changes / override the default implementation
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f'Argument {key} is not valid.')
            self.__setattr__(key, value)

        if self.exp_dep_params is None:
            self.exp_dep_params = self.default_exp_dep_params()

        if len(kwargs) > 0:
            self.update_params()

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

        for par_name in self.exp_indep_params:
            if self.is_amp_par(par_name):
                p = self.params[par_name]
                p.vary = True
                p.enabled = True

    def get_current_species_names(self):
        return self.spec_names

    def _get_traces(self, t, ks, j, i):
        raise NotImplementedError()


class VarOrderModel(_FixedParametersModel):

    name = 'Variable Order A→B model'

    def __init__(self, *args, **kwargs):
        exps_data = args[0]  # exps_data must be first
        spec_names = ['A', 'B']
        spec_visible = [{'A': True, 'B': False} for _ in range(len(exps_data))]
        kwargs.update(n_spec=2, spec_names=spec_names, spec_visible=spec_visible)
        super(VarOrderModel, self).__init__(*args, **kwargs)

    def get_all_param_names(self):
        pars = {f'_{name}_0': f'Initial concentration of {name}' for name in self.spec_names}
        pars.update({name: f'Amplitude of {name}' for name in self.spec_names})  # amplitudes
        pars.update({'k': "The rate constant"})
        pars.update({'n': "Order of the reaction"})
        pars.update({'intercept': 'Intercept'})  # intercept
        return pars

    def default_exp_dep_params(self):
        pars = self.spec_names[:]  # amplitudes  # make a copy!!!
        pars += ['intercept']  # intercepts
        return pars

    def get_param_dicts(self):
        rates_dict = {'k': 0}
        amps_dict = {name: i for i, name in enumerate(self.spec_names)}
        j_dict = {f'_{name}_0': i for i, name in enumerate(self.spec_names)}

        return dict(rates_dict=rates_dict, amps_dict=amps_dict, j_dict=j_dict)

    def add_params(self, param_set: list = None, dict_params: dict = None, j_dict: dict = None, rates_dict: dict = None,
                   amps_dict: dict = None, species_visible: [dict] = None, par_format=lambda name: name, **kwargs):

        for par in param_set:
            f_par_name = par_format(par)

            vary = True
            value = 1
            min = -np.inf
            max = np.inf
            if self.is_j_par(par):
                dict_params['j'][j_dict[par]] = f_par_name
                vary = False
                if par == '_A_0':
                    value = 1
                else:
                    value = 0
            elif par.startswith('k'):
                dict_params['rates'][rates_dict[par]] = f_par_name
                min = 0
            elif self.is_amp_par(par):
                dict_params['amps'][amps_dict[par]] = f_par_name
                vary = not self.varpro
            elif self.is_intercept(par):
                dict_params['intercept'] = f_par_name
                vary = self.fit_intercept_varpro and not self.varpro
                value = 0
            elif par.startswith('n'):
                value = 1
                vary = True
                min = 0
                max = 20
            else:
                value = 0

            dict_params['all'].append(f_par_name)
            if f_par_name not in self.params:
                self.params.add(f_par_name, min=min, max=max, value=value, vary=vary)
                # add an enabled attribute for each parameter
                self.params[f_par_name].enabled = True

    def _get_traces(self, t, ks, j, i):

        j_A, j_B = j
        k = ks[0]

        n_par = list(filter(lambda name: name.startswith('n'), self.param_names_dict[i]['all']))[0]
        n = self.params[n_par].value

        C = np.empty((t.shape[0], 2), dtype=np.float64)

        if n == 1:  # first order
            C[:, 0] = j_A * np.exp(-k * t)
        else:  # n-th order, this wont work for negative x values, c(t) = 1-n root of (c^(1-n) + k*(n-1)*t)
            expr_in_root = np.power(float(j_A), 1 - n) + k * (n - 1) * t
            expr_in_root = expr_in_root.clip(min=0)  # set to 0 all the negative values
            C[:, 0] = np.power(expr_in_root, 1.0 / (1 - n))

        C[:, 1] = j_B + j_A - C[:, 0]

        return np.nan_to_num(np.heaviside(t[:, None], 1) * C)


class Photosensitization(_FixedParametersModel):

    name = 'Photosensitization (PS→, PS+Q→T, T→) [Q]_0 \u226b [PS]_0'

    def __init__(self, *args, **kwargs):
        exps_data = args[0]  # exps_data must be first
        spec_names = ['PS', 'T']
        spec_visible = [{'PS': True, 'T': True} for _ in range(len(exps_data))]
        kwargs.update(n_spec=2, spec_names=spec_names, spec_visible=spec_visible)
        super(Photosensitization, self).__init__(*args, **kwargs)

    def get_all_param_names(self):
        pars = {'_Q_0': 'Concentration of a quencher'}
        pars.update({f'_{name}_0': f'Initial concentration of {name}' for name in self.spec_names})
        pars.update({name: f'Amplitude of {name}' for name in self.spec_names})  # amplitudes
        pars.update({'k_q': "Quenching rate constant"})
        pars.update({'k_PS': "Decay rate constant of a PS without a quencher"})
        pars.update({'k_T': "Decay rate constant of a sensitized triplet T"})
        pars.update({'intercept': 'Intercept'})  # intercept
        return pars

    def default_exp_dep_params(self):
        pars = ['intercept']  # intercepts
        pars += ['_Q_0']  # initial concentration of a quencher
        pars += self.spec_names  # amplitudes
        return pars

    def get_param_dicts(self):
        rates_dict = {name: i for i, name in enumerate(['k_q', 'k_PS', 'k_T'])}
        amps_dict = {name: i for i, name in enumerate(self.spec_names)}
        j_dict = {'_Q_0': 0}
        j_dict.update({f'_{name}_0': i+1 for i, name in enumerate(self.spec_names)})

        return dict(rates_dict=rates_dict, amps_dict=amps_dict, j_dict=j_dict)

    def add_params(self, param_set: list = None, dict_params: dict = None, j_dict: dict = None, rates_dict: dict = None,
                   amps_dict: dict = None, species_visible: [dict] = None, par_format=lambda name: name, **kwargs):

        for par in param_set:
            f_par_name = par_format(par)

            vary = True
            value = 1
            min = -np.inf
            if self.is_j_par(par):
                dict_params['j'][j_dict[par]] = f_par_name
                vary = False
                if par == '_PS_0' or par == '_Q_0':
                    value = 1
                else:
                    value = 0
            elif self.is_rate_par(par):
                dict_params['rates'][rates_dict[par]] = f_par_name
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

            dict_params['all'].append(f_par_name)
            if f_par_name not in self.params:
                self.params.add(f_par_name, min=min, max=np.inf, value=value, vary=vary)
                # add an enabled attribute for each parameter
                self.params[f_par_name].enabled = True

    def _get_traces(self, t, ks, j, i):

        assert ks.shape == j.shape

        # diff equations:
        #   dPS/dt = -k_PS * [PS] - k_q * [Q]*[PS]
        #   dT/dt = + k_q * [Q]*[PS] - k_T * [T]
        #
        #   solution for [Q]_0 >> [PS]_0
        #  c(PS) = [PS]_0 * exp(-(k_PS + k_q * [Q]) * t)
        #  c(T) = ... double exponential

        C = np.empty((t.shape[0], 2), dtype=np.float64)

        j_Q, j_PS, j_T = j
        k_q, k_PS, k_T = ks

        k_form = k_q * j_Q  # formation of T rate constant

        C[:, 0] = j_PS * np.exp(-t * (k_PS + k_form))
        C[:, 1] = j_T + (k_form / (k_T - k_form)) * (np.exp(-k_form * t) - np.exp(-k_T * t))

        return np.nan_to_num(np.heaviside(t[:, None], 1) * C)


class GeneralFitModel(_InterceptVarProModel):

    name = 'General model'

    def __init__(self, *args, show_backward_rates=False, **kwargs):
        super(GeneralFitModel, self).__init__(*args, **kwargs)

        self.general_model = GeneralModel()

        self.show_backward_rates = show_backward_rates

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

    def get_param_dicts(self):
        rate_vals = self.general_model.get_rates(self.show_backward_rates, True)
        rates_dict = {rate[0]: i for i, rate in enumerate(rate_vals)}
        amps_dict = {name: i for i, name in enumerate(self.spec_names)}
        j_dict = {f'_{name}_0': i for i, name in enumerate(self.spec_names)}

        return dict(rates_dict=rates_dict, amps_dict=amps_dict, j_dict=j_dict, rate_vals=rate_vals)

    def add_params(self, param_set: list = None, dict_params: dict = None, j_dict: dict = None, rates_dict: dict = None,
                   amps_dict: dict = None, species_visible: [dict] = None, par_format=lambda name: name, rate_vals=None):

        for par in param_set:
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

                f = list(filter(lambda r: par == r[0], rate_vals))  # get rates from general model
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

            dict_params['all'].append(f_par_name)
            if f_par_name not in self.params:
                self.params.add(f_par_name, min=min, max=np.inf, value=value, vary=vary)
                # add an enabled attribute for each parameter
                self.params[f_par_name].enabled = True

    def _get_traces(self, x_data, rates, j, i):
        if x_data[0] > 0:  # initial conditions are valid for time=0
            n = 100  # prepend x values with 100 points if not starting with zero time
            x_prepended = np.concatenate((np.linspace(0, x_data[0], n, endpoint=False), x_data))
            return odeint(self.general_model.func, j, x_prepended, args=(rates,))[n:, :]

        elif x_data[0] < 0:
            x_pos = x_data[x_data >= 0]  # find x >= 0
            sol = np.zeros((x_data.shape[0], j.shape[0]), dtype=np.float64)
            if x_pos.shape[0] > 1:  # simulate only for at least 2 positive values
                sol[(x_data < 0).sum():, :] = self._get_traces(x_pos, rates, j, i)  # use recursion here

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
