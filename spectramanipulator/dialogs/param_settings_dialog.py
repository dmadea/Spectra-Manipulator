from PyQt5 import QtWidgets

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QLabel
from spectramanipulator.dialogs.genericinputdialog import GenericInputDialog


class ParamSettingsDialog(GenericInputDialog):

    def __init__(self, species_names: list, exp_dep_params_names: list):

        if self.is_opened:
            return


        self.options =

        widget_list = []

        self.ftol_edit = QLineEdit(str(ftol))
        self.xtol_edit = QLineEdit(str(xtol))
        self.gtol_edit = QLineEdit(str(gtol))

        self.loss_cb = QComboBox()
        self.loss_cb.addItems(map(lambda o: f"{o['opt']}: {o['description']}", self.loss_options))
        _opts = list(map(lambda o: o['opt'], self.loss_options))
        self.loss_cb.setCurrentIndex(_opts.index(loss))

        self.max_nfev_edit = QLineEdit(str(max_nfev))

        self.verbose_cb = QComboBox()
        self.verbose_cb.addItems(map(lambda o: f"{o['opt']}: {o['description']}", self.verbose_opts))
        _opts = list(map(lambda o: o['opt'], self.verbose_opts))
        self.verbose_cb.setCurrentIndex(_opts.index(verbose))

        # widget_list.append(('Algorithm to perform minimization:', self.alg_cb))
        widget_list.append(('ftol:', self.ftol_edit))
        widget_list.append(('xtol:', self.xtol_edit))
        widget_list.append(('gtol:', self.gtol_edit))
        widget_list.append(('Loss function:', self.loss_cb))
        widget_list.append(('Max n_fev:', self.max_nfev_edit))
        widget_list.append(('Verbose:', self.verbose_cb))

        super(ParamSettingsDialog, self).__init__(widget_list=widget_list,
                                                  label_text='For explanations, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html.',
                                                  title='Settings of Trust Region Reflective algorithm',
                                                  parent=None,
                                                  set_result=set_result)

    def accept(self):
        # self.options['method'] = self.minim_alg_opts[self.alg_cb.currentIndex()]['opt']
        self.options['loss'] = self.loss_options[self.loss_cb.currentIndex()]['opt']
        self.options['verbose'] = self.verbose_opts[self.verbose_cb.currentIndex()]['opt']
        try:
            self.options['ftol'] = float(self.ftol_edit.text())
        except ValueError:
            self.options['ftol'] = None
        try:
            self.options['xtol'] = float(self.xtol_edit.text())
        except ValueError:
            self.options['xtol'] = None
        try:
            self.options['gtol'] = float(self.gtol_edit.text())
        except ValueError:
            self.options['gtol'] = None
        try:
            self.options['max_nfev'] = float(self.max_nfev_edit.text())
        except ValueError:
            self.options['max_nfev'] = None

        super(ParamSettingsDialog, self).accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = ParamSettingsDialog()
    Dialog.show()
    sys.exit(app.exec_())

