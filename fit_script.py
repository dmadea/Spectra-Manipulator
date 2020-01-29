from spectrum import Spectrum
import numpy as np

from user_namespace import add_to_list
from dialogs.fitwidget import FitResult
from scipy.integrate import odeint
from PyQt5.QtGui import QColor

import lmfit
from lmfit import fit_report, report_fit, Minimizer, report_ci, conf_interval, conf_interval2d, Parameters
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# import pyqtgraph as pg

from main import Main


def init_model(k0, kq, kd, kTT=1e4, kdel=1):
    params = Parameters()

    params.add('Xan', value=0, min=-np.inf, max=np.inf, vary=False)
    params.add('DR', value=0, min=-np.inf, max=np.inf, vary=False)
    params.add('DF', value=1, min=-np.inf, max=np.inf, vary=False)

    params.add('A0', value=0, min=0, max=10, vary=False)
    params.add('kex', value=10, min=1e-3, max=1e3, vary=True)
    params.add('k_ex', value=0, min=0, max=np.inf, vary=False)
    params.add('k0', value=k0, min=0, max=np.inf, vary=False)
    params.add('kq', value=kq, min=0, max=np.inf, vary=False)
    params.add('kd', value=kd, min=0, max=np.inf, vary=False)
    params.add('kTT', value=kTT, min=0, max=np.inf, vary=False)
    params.add('kdel', value=kdel, min=0.01, max=1e3, vary=True)
    params.add('ksq', value=0, min=0, max=1e8, vary=False)
    #
    # params.add('A00', value=0.003, min=0.001, max=3, vary=True)
    # params.add('A01', value=0.008, min=0.001, max=3, vary=True)
    # params.add('A02', value=0.015, min=0.001, max=3, vary=True)
    # params.add('A03', value=0.02, min=0.005, max=3, vary=True)
    # params.add('A04', value=0.02, min=0.005, max=3, vary=True)
    # params.add('A05', value=0.02, min=0.005, max=3, vary=True)
    # params.add('A06', value=0.03, min=0.005, max=3, vary=True)
    # params.add('A07', value=0.03, min=0.005, max=3, vary=True)
    # params.add('A08', value=0.04, min=0.005, max=3, vary=True)

    params.add('A00', value=1, min=0.001, max=3, vary=False)
    params.add('A01', value=1, min=0.001, max=3, vary=False)
    params.add('A02', value=1, min=0.001, max=3, vary=False)
    params.add('A03', value=1, min=0.005, max=3, vary=False)
    params.add('A04', value=1, min=0.005, max=3, vary=False)
    params.add('A05', value=1, min=0.005, max=3, vary=False)
    params.add('A06', value=1, min=0.005, max=3, vary=False)
    params.add('A07', value=1, min=0.005, max=3, vary=False)
    params.add('A08', value=1, min=0.005, max=3, vary=False)

    # params.add('cCP1', value=1e-4, min=0, max=np.inf, vary=False)
    # params.add('y0', value=0, min=-np.inf, max=np.inf, vary=False)

    return params


def get_fit_data(params, concentrations, times):
    Xan, DR, DF = params['Xan'].value, params['DR'].value, params['DF'].value
    A0, kex, k_ex, k0, kq, kd, kTT, kdel = params['A0'].value, params['kex'].value, params['k_ex'].value, params[
        'k0'].value, params['kq'].value, params['kd'].value, params['kTT'].value, params['kdel'].value

    ksq = params['ksq'].value

    A = []
    A.append(params['A00'].value)
    A.append(params['A01'].value)
    A.append(params['A02'].value)
    A.append(params['A03'].value)
    A.append(params['A04'].value)
    A.append(params['A05'].value)
    A.append(params['A06'].value)
    A.append(params['A07'].value)
    A.append(params['A08'].value)

    def Tall(A):
        result = (1 - np.power(10, -A)) / (A * np.log(10))
        result[np.isnan(result)] = 1
        return result

    def solve(profiles, t, cCP1):
        Xan, Ex, DR, DF = profiles

        dcXan_dt = -(k0 + kq * cCP1) * Xan
        dcEx_dt = + (kq - ksq) * cCP1 * Xan - (kex + kd) * Ex + k_ex * DR
        dcDR_dt = + kex * Ex - kd * DR - k_ex * DR
        # DF simulation
        dcDF_dt = kTT * Xan * DR - kdel * DF

        return [dcXan_dt, dcEx_dt, dcDR_dt, dcDF_dt]

    temp = []
    for i in range(len(times)):
        x0 = np.linspace(0, times[i][0], num=100)
        # k1 = concentrations[i] * kq

        cCP1 = concentrations[i]

        _init_x = odeint(solve, [1, 0, 0, 0], x0, args=(cCP1,))[-1, :]  # take the row in the result matrix
        result = odeint(solve, _init_x, times[i], args=(cCP1,))

        cXan = result[:, 0]
        cEx = result[:, 1]
        cDR = result[:, 2]
        cDF = result[:, 3] * (Tall(A0 * cXan) if A0 > 0 else 1)

        calc_DF = Xan * cXan + DR * (cEx + cDR) + A[i] * cDF / np.max(cDF)
        temp.append(calc_DF)

    return np.hstack(temp)


# def _get_fit_data()


def get_constants(T=-30, T_index=None):
    Tk = T + 273.15
    R = 8.314462618153

    k0Ea, k0A = 5.126043 * 1e3, np.exp(1.50257E+01)
    kqEa, kqA = 8.133688 * 1e3, np.exp(2.6645E+01)
    kdEa, kdA = 25.63632243 * 1e3, np.exp(22.52314)

    k0 = k0A * np.exp(-k0Ea / (R * Tk))
    kq = kqA * np.exp(-kqEa / (R * Tk))
    kd = kdA * np.exp(-kdEa / (R * Tk))

    # -30, -20, -10, ..., 50 °C
    k0s = [0.271421, 0.295687, 0.321012, 0.346565, 0.373799, 0.398697, 0.432975, 0.473055, 0.515757]
    kqs = [6777.76, 7626.51, 8997.04, 10296.96, 12014.43, 13440.21, 14935.50, 16455.38, 17680.98]
    kds = [0.01608260, 0.03010330, 0.05319809, 0.08819585, 0.13405819, 0.19101218, 0.25347676, 0.31607057, 0.39165471]

    if T_index is not None:
        return k0s[T_index], kqs[T_index], kds[T_index]
    else:
        return k0 * 1e-6, kq * 1e-6, kd * 1e-6


def fit(item, g_idx=0, T=-30, t_min=0.2, method='leastsq', params=None, res_shift=-0.1):
    group = item[g_idx]
    n = group.__len__()

    T_index = int((T + 30) * 8 / 80)

    # t_index * 10 - 30 = T

    times = []
    conc_data = []

    for i in range(n):
        start_idx = Spectrum.find_nearest_idx(group[i].data[:, 0], t_min)
        conc_data.append(group[i].data[start_idx:, 1])
        times.append(group[i].data[start_idx:, 0])

    raw_data = np.hstack(conc_data)

    k0, kq, kd = get_constants(T, T_index)

    # k0, kq, kd = 0.128296, 12423, 0.19101218  # p=0.9 Xan conc.
    # k0, kq, kd = 0.177918922, 13357, 0.19101218  # p=1.5 Xan conc.

    params = params if params is not None else init_model(k0, kq, kd)

    def residuals(params):
        fit_data = get_fit_data(params, [float(group[k].name) for k in range(n)], times)

        res = fit_data - raw_data

        return res

    minimizer = Minimizer(residuals, params)
    result = minimizer.minimize(method=method)  # fit

    fit_data = get_fit_data(result.params, [float(group[k].name) for k in range(n)], times)

    x_val = np.linspace(0, fit_data.shape[0] - 1, fit_data.shape[0])

    sp_raw = Spectrum.from_xy_values(x_val, raw_data, name=f'concatenated data-{group.name}',
                                     line_width=0.2)  # , line_type=0, symbol='o', symbol_size=1)
    sp_fit = Spectrum.from_xy_values(x_val, fit_data, name=f'fit-{group.name}', color='black', line_width=None)
    sp_res = Spectrum.from_xy_values(x_val, fit_data - raw_data - res_shift, name=f'res-{group.name}',
                                     color=(100, 100, 100, 255), line_width=0.2)

    report_fit(result)

    values_errors = np.zeros((2, 2))
    values_errors[0, 0] = result.params['kex'].value
    values_errors[0, 1] = result.params['kex'].stderr
    values_errors[1, 0] = result.params['kdel'].value
    values_errors[1, 1] = result.params['kdel'].stderr

    fit_result = FitResult(result, minimizer, values_errors, None)

    add_to_list(sp_raw)
    add_to_list(sp_fit)
    add_to_list(sp_res)

    return fit_result, result.params




def plot_traces(item, indices, xlims=(-0.3, 18), ylims=(-0.02, 0.25), figsize=(15, 15), dpi=500, lw=0.2):
    if not isinstance(indices, (list, tuple, np.ndarray)):
        return

    r = len(indices)

    if r == 0:
        return

    c = len(item[indices[0]])

    plt.clf()

    fig, ax = plt.subplots(r, c, figsize=figsize, gridspec_kw={'hspace': 0, 'wspace': 0})

    for i in range(r):
        curr_gr = item[indices[i]]

        for j in range(c):

            x = curr_gr[j].data[:, 0]
            y = curr_gr[j].data[:, 1]

            if i == 0:
                ax[i, j].set_title(f"[CP1]={float(curr_gr[j].name):.2e} $M$")

            if i == r - 1:
                ax[i, j].set_xlabel('Time ($\mu s$)')

            qc = Main.intColor(j * c + i, hues=r, values=c, minHue=0, maxHue=320, reversed=True)
            color = (qc.red() / 255.0, qc.green() / 255.0, qc.blue() / 255.0)

            ax[i, j].plot(x, y, lw=lw, color=color)

            # ax[i, j].legend()
            if j == 0:
                ax[i, j].set_ylabel(f"{curr_gr.name} °C")
            if xlims:
                ax[i, j].set_xlim(xlims[0], xlims[1])
            if ylims:
                ax[i, j].set_ylim(ylims[0], ylims[1])

            if j > 0:
                ax[i, j].yaxis.set_ticks([])

            if i < r - 1:
                ax[i, j].xaxis.set_ticks([])
            else:
                ax[i, j].xaxis.set_ticks([0, 2, 4, 6, 8, 10, 13, 16])


    plt.tight_layout()
    # plt.show()

    plt.savefig(fname=r'C:\Users\Dominik\Desktop\snth\matrix.pdf', format='pdf', transparent=True, dpi=dpi)
