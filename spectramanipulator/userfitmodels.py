from .fitmodels import _Model
from scipy.integrate import odeint
import numpy as np


#
# # uncomment this section and change name of the class of new model
# class NewModel(_Model):
#
#     # change name of the model here
#     name = 'User defined model'
#
#     # add corresponding lines to this method for each variable that this model contains
#     # setup the initial value, minimum, maximum and initial option if the variable
#     # will be varied during fitting
#     # use np.inf for positive infinity and -np.inf for negative infinity
#     def init_params(self):
#         self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('C', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('D', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('k2', value=0.5, min=0, max=np.inf, vary=False)
#         self.params.add('k3', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('k4', value=1, min=0, max=np.inf, vary=True)
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     # change the procedure to how to initialize the values of your parameters from the selected user data
#     # the x_data is array of x values region of fitted data and y_data is array of y values of fitted data
#     # this is not mandatory, if you want, you can omit the initilization, it this case, just delete all lines
#     # and keep only 'pass'
#     def initialize_values(self, x_data, y_data):
#         self.params['A'].value = np.max(y_data[0])
#         # self.params['B'].value = y_data[0]
#
#         pass
#
#     # change this function, the order of the parameters in the func method must be the same as was defined in
#     # init_params method!, first is parameter x, which are the x values of fitted data, then the parameters.
#     # the function has to always return numpy array that is the same dimension as x
#     # for solving differential equation, use odeint method
#     @staticmethod
#     def func(x, c0, A, B, C, D, k1, k2, k3, y0):
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

class DF(_Model):
    """l"""

    name = 'DF plot'

    def init_params(self):
        self.params.add('Xan', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('DR', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('DF', value=1, min=-np.inf, max=np.inf, vary=True)

        self.params.add('A0', value=0, min=0, max=10, vary=False)
        self.params.add('kex', value=1.5, min=0, max=np.inf, vary=True)
        self.params.add('k_ex', value=0, min=0, max=np.inf, vary=False)
        self.params.add('k0', value=0.39, min=0, max=np.inf, vary=False)
        self.params.add('kq', value=1.34e4, min=0, max=np.inf, vary=False)
        self.params.add('kd', value=0.186, min=0, max=np.inf, vary=False)
        self.params.add('kTT', value=1e4, min=0, max=np.inf, vary=False)
        self.params.add('kdel', value=1e4, min=0, max=np.inf, vary=True)
        self.params.add('cCP1', value=1e-4, min=0, max=np.inf, vary=False)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=False)

    def initialize_values(self, x_data, y_data):
        # self.params['A0'].value = np.max(y_data[0])
        # self.params['B'].value = y_data[0]
        pass

    @staticmethod
    def func(x, Xan, DR, DF, A0, kex, k_ex, k0, kq, kd, kTT, kdel, cCP1, y0):
        def Tall(A):
            result = (1 - np.power(10, -A)) / (A * np.log(10))
            result[np.isnan(result)] = 1
            return result

        def solve(profiles, t, *args):
            Xan, Ex, DR, DF = profiles
            k1_var = args[0]

            dcXan_dt = -(k0 + k1_var) * Xan
            dcEx_dt = + k1_var * Xan - (kex + kd) * Ex + k_ex * DR
            dcDR_dt = + kex * Ex - kd * DR - k_ex * DR
            # DF simulation
            dcDF_dt = kTT * Xan * DR - kdel * DF

            return [dcXan_dt, dcEx_dt, dcDR_dt, dcDF_dt]

        x0 = np.linspace(0, x[0], num=100)
        _init_x = odeint(solve, [1, 0, 0, 0], x0, args=(kq * cCP1,))[-1, :]  # take the row in the result matrix

        result = odeint(solve, _init_x, x, args=(kq * cCP1,))
        cXan = result[:, 0]
        cEx = result[:, 1]
        cDR = result[:, 2]
        cDF = result[:, 3] * (Tall(A0 * cXan) if A0 > 0 else 1)

        return Xan * cXan + DR * (cEx + cDR) + DF * cDF / np.max(cDF) + y0


class DF_max(_Model):
    """l"""

    name = 'DF Maxima'

    def init_params(self):
        self.params.add('A0', value=0, min=0, max=10, vary=False)
        self.params.add('kex', value=1.5, min=0, max=np.inf, vary=True)
        self.params.add('k_ex', value=0, min=0, max=np.inf, vary=False)
        self.params.add('k0', value=0.128296, min=0, max=np.inf, vary=False)
        self.params.add('kq', value=13440.21, min=0, max=np.inf, vary=False)
        self.params.add('kd', value=0.19101218, min=0, max=np.inf, vary=False)
        self.params.add('kTT', value=1e4, min=0, max=np.inf, vary=False)
        self.params.add('kdel', value=1e4, min=0, max=np.inf, vary=True)
        self.params.add('ksq', value=0, min=0, max=np.inf, vary=True)
        self.params.add('points', value=500, min=1, max=np.inf, vary=False)
        self.params.add('t_lim', value=6, min=0, max=np.inf, vary=False)
        self.params.add('T', value=20, min=-30, max=50, vary=False)

    def initialize_values(self, x_data, y_data):
        # self.params['A0'].value = np.max(y_data[0])
        # self.params['B'].value = y_data[0]
        pass

    @staticmethod
    def func(x, A0, kex, k_ex, k0, kq, kd, kTT, kdel, ksq, points, t_lim, T):
        intensity = False

        # from fit_script import get_constants
        #
        # T_index = int((T + 30) * 8 / 80)
        # k0, kq, kd = get_constants(T=T, T_index=T_index)

        def Tall(A):
            result = (1 - np.power(10, -A)) / (A * np.log(10))
            result[np.isnan(result)] = 1
            return result

        def solve(profiles, t, *args):
            Xan, Ex, DR, DF = profiles
            cCP1 = args[0]

            dcXan_dt = -(k0 + kq * cCP1 + ksq * cCP1) * Xan
            dcEx_dt = + kq * cCP1 * Xan - (kex + kd) * Ex + k_ex * DR
            dcDR_dt = + kex * Ex - kd * DR - k_ex * DR
            # DF simulation
            dcDF_dt = kTT * Xan * DR - kdel * DF

            return [dcXan_dt, dcEx_dt, dcDR_dt, dcDF_dt]

        def integrate(k1, times, init_values):
            result = odeint(solve, init_values, times, args=(k1,))
            Xan = result[:, 0]
            Ex = result[:, 1]
            DR = result[:, 2]
            DF = result[:, 3]
            return Xan, Ex, DR, DF

        def max(k1):
            limit_t = t_lim
            num_p = points
            t = np.linspace(0, limit_t, num_p)

            Xan, Ex, DR, DF = integrate(k1, t, [1, 0, 0, 0])
            DF_corr = DF * (Tall(A0 * Xan) if A0 > 0 else 1)
            t_max_idx = np.argmax(DF_corr)

            min_idx = t_max_idx - 1 if t_max_idx > 0 else 0
            t_part = np.linspace(t[min_idx], t[t_max_idx + 1], num_p)
            Xan, Ex, DR, DF = integrate(k1, t_part, [Xan[min_idx], Ex[min_idx], DR[min_idx], DF[min_idx]])

            DF_corr = DF * (Tall(A0 * Xan) if A0 > 0 else 1)

            return t_part[np.argmax(DF_corr)] if not intensity else np.max(DF_corr)

        maxima = np.zeros(x.shape[0])

        for i, k1 in enumerate(x):
            maxima[i] = max(k1)

        return maxima


#
# class DelayedFluorescence2(_Model):
#     """l"""
#
#     name = 'Delayed Fluorescence2'
#
#     def init_params(self):
#         # self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('Xan', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('DR', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('DF', value=1, min=-np.inf, max=np.inf, vary=True)
#         self.params.add('A0', value=0.2, min=-np.inf, max=np.inf, vary=False)
#
#         # self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
#         # self.params.add('C', value=0, min=-np.inf, max=np.inf, vary=False)
#         self.params.add('k0', value=0.4, min=0, max=np.inf, vary=False)
#         self.params.add('k1', value=1, min=0, max=np.inf, vary=False)
#         self.params.add('kd', value=0.186, min=0, max=np.inf, vary=False)
#         self.params.add('kex', value=0.5, min=0, max=np.inf, vary=True)
#
#         # self.params.add('k3', value=1, min=0, max=np.inf, vary=True)
#         # self.params.add('k4', value=1, min=0, max=np.inf, vary=True)
#
#         self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)
#
#     def initialize_values(self, x_data, y_data):
#         # self.params['A'].value = np.max(y_data[0])
#         # self.params['B'].value = y_data[0]
#         pass
#
#     # @nb.njit(fastmath=True)
#     # @staticmethod
#     # def sum(n=1000, A0=0):
#     #     sum = 0
#     #     for i in range(0, n):
#     #         sum += np.exp(-i / (n - 1) * A0)
#
#     @staticmethod
#     def func(x, Xan, DR, DF, A0, k0, k1, kd, kex, y0):
#
#         def Tall(A, classical=False):
#             if classical:
#                 return np.power(10, -A)
#             else:
#                 result = (1 - np.power(10, -A)) / (A * np.log(10))
#                 result[np.isnan(result)] = 1
#                 return result
#
#         def solve(conc, t):
#             cA, cB, cC = conc
#
#             dcA_dt = -(k0 + k1) * cA
#             dcB_dt = + k1 * cA - kex * cB - kd * cB
#             dcC_dt = + kex * cB - kd * cC
#
#             return [dcA_dt, dcB_dt, dcC_dt]
#
#         x0 = np.linspace(0, x[0], num=100)
#         _init_x = odeint(solve, [1, 0, 0], x0)[-1, :]  # take the row in the result matrix
#
#         result = odeint(solve, _init_x, x)
#         cXan = result[:, 0]
#         cEx = result[:, 1]
#         cDR = result[:, 2]
#         cDF = cXan * cDR
#
#         cDF *= Tall(cXan * A0) if A0 > 0 else 1  # selfabsorption correction
#
#         return Xan * cXan + DR * (cEx + cDR) + DF * cDF / np.max(cDF) + y0


class ABC1st_XanModel(_Model):
    """This is the A→B→C model, 1st order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] - k_2[\mathrm{B}]`

    :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = 0`
    """

    name = 'A→B→C (1st order-Xan-k0)'

    def init_params(self):
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('C', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k0', value=0.4, min=0, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]
        self.params['B'].value = y_data[0]

    @staticmethod
    def func(x, A, B, C, k0, k1, k2, y0):
        cA = np.exp(-(k0 + k1) * x)

        # if k1 == k2, the integrated law fow cB has to be changed
        if np.abs(k1 - k2) < 1e-10:
            cB = (k1 * (np.exp(k0 * x) - 1) * np.exp(x * (-(k0 + k1)))) / k0
        else:  # we would get zero division error for k1 = k2
            cB = (k1 * (np.exp(-k2 * x) - np.exp(-x * (k0 + k1)))) / (k0 + k1 - k2)

        cC = 1 - cA - cB

        return A * cA + B * cB + C * cC + y0


class Equlibrium(_Model):
    """This is the A→B→C model, 1st order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] - k_2[\mathrm{B}]`

    :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = 0`
    """

    name = 'Equilibrium model A+2L=2B, A+L=B and A+2L=B'

    def init_params(self):
        self.params.add('c0', value=1, min=0, max=np.inf, vary=True)
        self.params.add('epsA', value=1000, min=0, max=np.inf, vary=True)
        self.params.add('epsB', value=1000, min=0, max=np.inf, vary=True)
        self.params.add('K', value=2, min=0, max=np.inf, vary=True)
        self.params.add('model', value=0, min=0, max=2, vary=False)

    # def initialize_values(self, x_data, y_data):
    #     self.params['A'].value = y_data[0]
    #     self.params['B'].value = y_data[0]

    @staticmethod
    def func(x, c0, epsA, epsB, K, model):
        cL = x

        if model == 0:
            alpha = cL * (-K * cL + np.sqrt(K * (K * cL * cL + 16 * c0))) / (8 * c0)

            # alpha = np.zeros(cL.shape[0])
            # for i, cLi in enumerate(cL):
            #     polyi = [4*K*c0*c0, -4*K*c0*c0 - 4*K*c0*cLi + 4*c0, 4*K*c0*cLi + K*cLi*cLi, -K*cLi*cLi]
            #     roots = np.roots(polyi)  # find roots of this polynomial
            #     alphai = roots[roots >= 0][0]  # take first positive or zero root
            #
            #     alpha[i] = alphai

        elif model == 1:
            alpha = K * cL / (K * cL + 1)
        else:
            alpha = cL * (-K * cL + np.sqrt(K * (K * cL * cL + 4 * c0))) / (2 * c0)

            # alpha = K * cL * cL / (K * cL * cL + 1)

        return c0 * (epsA + alpha * ((2 if model == 0 else 1) * epsB - epsA))


if __name__ == '__main__':
    # m = LinearModel()

    import matplotlib.pyplot as plt

    # print(m.wrapper_func(np.asarray([0, 0.1, 0.15, 0.3, 0.6]), m.params))

    x = np.linspace(0, 10, 1000)

    m = DelayedFluorescence2()

    m.params['A'].value = 1
    m.params['A0cA'].value = 1
    m.params['k0'].value = 0.3

    m.params['k1'].value = 1.95
    m.params['kd'].value = 0.2

    n = DelayedFluorescence2()

    n.params['A'].value = 1
    n.params['A0cA'].value = 0
    n.params['k0'].value = 0.3
    n.params['k1'].value = 1.95
    n.params['kd'].value = 0.2

    mr = m.wrapper_func(x, m.params)
    nr = n.wrapper_func(x, n.params)

    plt.plot(x, mr, label='DelayedFluorescence2 A0 1')
    plt.plot(x, nr, label='DelayedFluorescence2 A0 0')

    plt.legend()
    plt.show()
