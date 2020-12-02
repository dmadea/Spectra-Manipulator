# from abc import abstractmethod
from lmfit import Parameters
import numpy as np
import inspect
from scipy.integrate import odeint


# import numba as nb


# import matplotlib.pyplot as plt
# from sympy import Derivative, symbols, Function, Equality
# from console import Console


# abstract class for specific model
class _Model(object):
    name = '[model name]'

    def __init__(self):
        self.params = Parameters()
        self.init_params()

    # @abstractmethod
    def init_params(self):
        pass

    # @abstractmethod
    def initialize_values(self, x_data, y_data):
        pass

    # @abstractmethod
    @staticmethod
    def func(x, *params):
        pass

    def wrapper_func(self, x, params):
        """Gets values from lmfit parameters and passes them into function func."""
        par_tup = (p.value for p in params.values())
        return self.func(x, *par_tup)

    def get_func_string(self) -> str:
        """Returns the implementation of the specific model func as a string."""
        text = inspect.getsource(self.func)

        it = iter(text.splitlines(keepends=True))
        # skip first two lines
        it.__next__()
        it.__next__()

        buffer = ""
        # 'remove' two tabs = 8 spaces from each line
        for line in it:
            buffer += line[8:]

        return buffer

    def par_count(self) -> int:
        """Returns the number of parameters for this model."""
        return self.params.__len__()


class AB1stModel(_Model):
    """This is the A→B model, 1st order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k[\mathrm{A}]`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k[\mathrm{A}]`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = 0`
    """

    name = 'A→B (1st order)'

    def init_params(self):
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k', value=1, min=0, max=np.inf, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]

    @staticmethod
    def func(x, A, B, k, y0):
        cA = np.exp(-k * x)
        return A * cA + B * (1 - cA) + y0


class ABVarOrderModel(_Model):
    """This is the A→B model, variable order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k[\mathrm{A}]^n`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k[\mathrm{A}]^n`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = 0`
    """

    name = 'A→B (var order)'

    def init_params(self):
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k', value=1, min=0, max=np.inf, vary=True)
        self.params.add('n', value=1, min=0, max=100, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]

    @staticmethod
    def func(x, c0, A, B, k, n, y0):
        if n == 1:  # first order
            cA = c0 * np.exp(-k * x)
        else:  # n-th order, this wont work for negative x values, c(t) = 1-n root of (c^(1-n) + k*(n-1)*t)
            expr_in_root = np.power(float(c0), 1 - n) + k * (n - 1) * x
            expr_in_root = expr_in_root.clip(min=0)  # set to 0 all the negative values
            cA = np.power(expr_in_root, 1.0 / (1 - n))

        return A * cA + B * (1 - cA) + y0


class AB_mixed12Model(_Model):
    """This is the A→B model, mixed 1st and 2nd order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}] - k_2[\mathrm{A}]^2`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] + k_2[\mathrm{A}]^2`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = 0`
    """

    name = 'A→B (Mixed 1st and 2nd order)'

    def init_params(self):
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]

    @staticmethod
    def func(x, c0, A, B, k1, k2, y0):
        cA = c0 * k1 / ((c0 * k2 + k1) * np.exp(k1 * x) - c0 * k2)
        return A * cA + B * (1 - cA) + y0


class ABC1stModel(_Model):
    """This is the A→B→C model, 1st order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] - k_2[\mathrm{B}]`

    :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = 0`
    """

    name = 'A→B→C (1st order)'

    def init_params(self):
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('C', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]
        self.params['B'].value = y_data[0]

    @staticmethod
    def func(x, A, B, C, k1, k2, y0):
        cA = np.exp(-k1 * x)

        # if k1 == k2, the integrated law fow cB has to be changed
        if np.abs(k1 - k2) < 1e-10:
            cB = k1 * x * np.exp(-k1 * x)
        else:  # we would get zero division error for k1 = k2
            cB = (k1 / (k2 - k1)) * (np.exp(-k1 * x) - np.exp(-k2 * x))

        cC = 1 - cA - cB

        return A * cA + B * cB + C * cC + y0


class ABCvarOrderModel(_Model):
    """This is the A→B→C model, variable order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]^{n_1}`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}]^{n_1} - k_2[\mathrm{B}]^{n_2}`

    :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]^{n_2}`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = 0`

    Note, that :math:`n_2` must be :math:`\ge 1`, however, :math:`n_1 \ge 0`.
    """

    name = 'A→B→C (var order)'

    def init_params(self):
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('C', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('n1', value=1, min=0, max=100, vary=False)
        # numerical integration won't work for n < 1, therefore, min=1
        self.params.add('n2', value=1, min=1, max=100, vary=False)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]
        self.params['B'].value = y_data[0]

    @staticmethod
    def func(x, c0, A, B, C, k1, k2, n1, n2, y0):
        def cA(c0, k, n, x):
            if n == 1:
                return c0 * np.exp(-k * x)
            else:
                expr_in_root = c0 ** (1 - n) + k * (n - 1) * x
                # we have to set all negative values to 0 (this is important for n < 1),
                # because the root of negative value would cause error
                if type(expr_in_root) is float:
                    expr_in_root = 0 if expr_in_root < 0 else expr_in_root
                else:
                    expr_in_root = expr_in_root.clip(min=0)
                return np.power(expr_in_root, 1.0 / (1 - n))

        # definition of differential equations
        def dB_dt(cB, x):
            return k1 * cA(c0, k1, n1, x) ** n1 - k2 * cB ** n2

        # solve for cB, cA is integrated law, initial conditions, cB(t=0)=0
        # but initial point for odeint must be defined for x[0]
        # evolve from 0 to x[0] and then take the last point as the initial condition
        x0 = np.linspace(0, x[0], num=100)
        init_B = odeint(dB_dt, 0, x0).flatten()[-1]
        result = odeint(dB_dt, init_B, x)

        cA = cA(c0, k1, n1, x)
        cB = result.flatten()
        cC = c0 - cA - cB

        return A * cA + B * cB + C * cC + y0


class ABCD1stModel(_Model):
    """This is the A→B→C→D model, 1st order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}] - k_2[\mathrm{B}]`

    :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}] - k_3[\mathrm{C}]`

    :math:`\\frac{\mathrm{d}[\mathrm{D}]}{\mathrm{d}t} = +k_3[\mathrm{C}]`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = [\mathrm D]_0 = 0`
    """
    name = 'A→B→C→D (1st order)'

    def init_params(self):
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('C', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('D', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('k3', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]
        self.params['B'].value = y_data[0]
        self.params['C'].value = y_data[0]

    @staticmethod
    def func(x, A, B, C, D, k1, k2, k3, y0):
        def concB(k1, k2, x):
            # if k1 == k2, the integrated law fow cB has to be changed
            if np.abs(k1 - k2) < 1e-10:
                return k1 * x * np.exp(-k1 * x)
            else:  # we would get zero division error for k1 = k2
                return (k1 / (k2 - k1)) * (np.exp(-k1 * x) - np.exp(-k2 * x))

        def dC_dt(cC, t):
            cB = concB(k1, k2, t)
            return k2 * cB - k3 * cC  # d[C]/dt = k2[B] - k3[C]

        # numerically integrate, initial condition, cC(t=0) = 0
        # but initial point for odeint must be defined for x[0]
        # evolve from 0 to x[0] and then take the last point as the initial condition
        x0 = np.linspace(0, x[0], num=100)
        init_C = odeint(dC_dt, 0, x0).flatten()[-1]
        result = odeint(dC_dt, init_C, x)

        cA = np.exp(-k1 * x)
        cB = concB(k1, k2, x)
        cC = result.flatten()
        cD = 1 - cA - cB - cC

        return A * cA + B * cB + C * cC + D * cD + y0


class ABCDvarOrderModel(_Model):
    """This is the A→B→C→D model, variable order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]^{n_1}`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}]^{n_1} - k_2[\mathrm{B}]^{n_2}`

    :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = +k_2[\mathrm{B}]^{n_2} - k_3[\mathrm{C}]^{n_3}`

    :math:`\\frac{\mathrm{d}[\mathrm{D}]}{\mathrm{d}t} = +k_3[\mathrm{C}]^{n_3}`

    initial conditions: :math:`[\mathrm A]_0 = c_0 \qquad [\mathrm B]_0 = [\mathrm C]_0 = [\mathrm D]_0 = 0`

    Note, that :math:`n_2,n_3` must be :math:`\ge 1`, however, :math:`n_1 \ge 0`.
    """
    name = 'A→B→C→D (var order)'

    def init_params(self):
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('C', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('D', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('k3', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('n1', value=1, min=0, max=100, vary=False)
        # numerical integration won't work for n < 1, therefore, min=1
        self.params.add('n2', value=1, min=1, max=100, vary=False)
        self.params.add('n3', value=1, min=1, max=100, vary=False)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]
        self.params['B'].value = y_data[0]
        self.params['C'].value = y_data[0]

    @staticmethod
    def func(x, c0, A, B, C, D, k1, k2, k3, n1, n2, n3, y0):
        def cA(c0, k, n, x):
            if n == 1:
                return c0 * np.exp(-k * x)
            else:
                expr_in_root = c0 ** (1 - n) + k * (n - 1) * x
                # we have to set all negative values to 0 (this is important for n < 1),
                # because the root of negative value would cause error
                if type(expr_in_root) is float:
                    expr_in_root = 0 if expr_in_root < 0 else expr_in_root
                else:
                    expr_in_root = expr_in_root.clip(min=0)
                return np.power(expr_in_root, 1.0 / (1 - n))

        # definition of differential equations
        def solve(concentrations, x):
            cB, cC = concentrations
            dB_dt = k1 * cA(c0, k1, n1, x) ** n1 - k2 * cB ** n2
            dC_dt = k2 * cB ** n2 - k3 * cC ** n3
            return [dB_dt, dC_dt]

        # solve for cB and cC, cA is integrated law, initial conditions, cB(t=0)=cC(t=0)=0
        # but initial point for odeint must be defined for x[0]
        # evolve from 0 to x[0] and then take the last points as the initial condition
        x0 = np.linspace(0, x[0], num=100)
        init_BC = odeint(solve, [0, 0], x0)[-1, :]  # take the last two points in the result matrix
        result = odeint(solve, init_BC, x)

        cA = cA(c0, k1, n1, x)
        cB = result[:, 0]
        cC = result[:, 1]
        cD = c0 - cA - cB - cC

        return A * cA + B * cB + C * cC + D * cD + y0


class ABCD_parModel(_Model):
    """This is the A→B, C→D parallel model, 1st order kinetics. Differential equations:

    :math:`\\frac{\mathrm{d}[\mathrm{A}]}{\mathrm{d}t} = -k_1[\mathrm{A}]`

    :math:`\\frac{\mathrm{d}[\mathrm{B}]}{\mathrm{d}t} = +k_1[\mathrm{A}]`

    :math:`\\frac{\mathrm{d}[\mathrm{C}]}{\mathrm{d}t} = -k_2[\mathrm{C}]`

    :math:`\\frac{\mathrm{d}[\mathrm{D}]}{\mathrm{d}t} = +k_2[\mathrm{C}]`

    initial conditions are same as in A→B model, since it is a 1st order, :math:`c_0` was not added to
    the parameters.
    """
    name = 'A→B, C→D (1st order)'

    def init_params(self):
        self.params.add('A', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('B', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('C', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('D', value=0, min=-np.inf, max=np.inf, vary=False)
        self.params.add('k1', value=1, min=0, max=np.inf, vary=True)
        self.params.add('k2', value=0.5, min=0, max=np.inf, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A'].value = y_data[0]
        self.params['C'].value = y_data[0]

    @staticmethod
    def func(x, A, B, C, D, k1, k2, y0):
        cA = np.exp(-k1 * x)
        cC = np.exp(-k2 * x)

        return A * cA + B * (1 - cA) + C * cC + D * (1 - cC) + y0


class LinearModel(_Model):
    """This is the linear model,
    :math:`y(x) = ax + y_0`
    where :math:`a` is a slope and :math:`y_0` intercept.
    """
    name = 'Linear'

    def init_params(self):
        self.params.add('a', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('y0', value=1, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['a'].value = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
        self.params['y0'].value = y_data[0]

    @staticmethod
    def func(x, a, y0):
        return a * x + y0


class Photobleaching_Model(_Model):
    """This is the photobleaching model. Differential equation:

    :math:`\\frac{\mathrm{d}c(t)}{\mathrm{d}t} = -k\\left(1 - 10^{-\\varepsilon c(t)l}\\right)`

    where :math:`k = \\frac{1}{V}I_0\Phi` is a product of incident light intensity and photoreaction quantum yield
    divided by the volume of the solution.

    initial conditions: :math:`c(0) = c_0 = \\frac{A_0}{\\varepsilon l}`

    Integrated law for evolution of absorbance in time (fitting equation):

    :math:`A(t) = \\frac{1}{\\ln 10}\\ln\\left(  10^{-A_0k^{\\prime}t}  \\left( 10^{A_0} + 10^{A_0k^{\\prime}t} - 1  \\right)\\right)`

    where :math:`A_0` is absorbance at time 0, :math:`k^{\\prime} = \\frac{k}{c_0}`, both :math:`k` and :math:`c_0`
    are fitting parameters, one of them must be fixed, usually :math:`c_0`
    """
    name = 'Photobleaching'

    def init_params(self):
        self.params.add('c0', value=1, min=0, max=np.inf, vary=False)
        self.params.add('A0', value=1, min=-np.inf, max=np.inf, vary=True)
        self.params.add('k', value=1, min=0, max=np.inf, vary=True)
        self.params.add('y0', value=0, min=-np.inf, max=np.inf, vary=True)

    def initialize_values(self, x_data, y_data):
        self.params['A0'].value = y_data[0]

    @staticmethod
    def func(x, c0, A0, k, y0):
        kp = k / c0  # k'
        return np.log(10 ** (-A0 * kp * x) * (10 ** A0 + 10 ** (A0 * kp * x) - 1)) / np.log(10) + y0


class Eyring(_Model):
    """This is the Eyring model,
    :math:`k(T) = \\frac{\\kappa k_B T}{h}e^{\\frac{\\Delta S}{R}}e^{-\\frac{\\Delta H}{RT}}`
    """
    name = 'Eyring'

    def init_params(self):
        self.params.add('dH', value=1000, min=-np.inf, max=np.inf, vary=True)
        self.params.add('dS', value=100, min=-np.inf, max=np.inf, vary=True)
        self.params.add('kappa', value=1, min=0, max=1, vary=False)

    def initialize_values(self, x_data, y_data):
        # self.params['a'].value = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
        # self.params['y0'].value = y_data[0]
        pass

    @staticmethod
    def func(x, dH, dS, kappa):
        kB = 1.38064852 * 1e-23  # boltzmann
        h = 6.62607004 * 1e-34  # planck
        R = 8.31446261815324  # universal gas constant
        # x - thermodynamic temperature
        return (kappa * kB * x / h) * np.exp((dS * x - dH) / (R * x))
