from ..qt_task import Task
from ..logger import Logger
import time
from lmfit import Minimizer
# from multiprocessing import Process
import numpy as np
from ..dialogs.fitresult import FitResult


class TaskFit(Task):
    def __init__(self, fw, model, method='least_squares', model_options=None, parent=None):
        super(TaskFit, self).__init__(parent)
        self.fw = fw  # fit widget
        self.model = model
        self.method = method
        self.model_options = {} if model_options is None else model_options

        self.result = None
        self.minimizer = None

    def preRun(self):
        self.fw.btnFit.setText('Cancel')

    def run(self):
        Logger.debug('Fitting...')
        self.run_fit()

    def run_fit(self):

        start_time = time.perf_counter()

        self.minimizer = Minimizer(self.model.residuals, self.model.params,
                                   iter_cb=lambda params, it, resid, *args, **kws: self.isInterruptionRequested())

        self.result = self.minimizer.minimize(method=self.method, **self.model_options)  # fit

        end_time = time.perf_counter()
        Logger.debug(end_time - start_time, 's for fitting')

    def postRun(self):  # after run has finished
        values_errors = np.zeros((len(self.result.params), 2), dtype=np.float64)
        for i, (p, new_p) in enumerate(zip(self.model.params.values(), self.result.params.values())):
            p.value = new_p.value  # update fitted parameters
            values_errors[i, 0] = p.value
            values_errors[i, 1] = p.stderr if p.stderr is not None else 0

        x_vals, fits, residuals = self.model.simulate()

        self.fw.setup_fields()
        self.fw.plot_fits(x_vals, fits, residuals)

        self.fw.fit_result = FitResult(self.result, self.minimizer, values_errors, self.model)
        self.fw.btnFit.setText('Fit')


