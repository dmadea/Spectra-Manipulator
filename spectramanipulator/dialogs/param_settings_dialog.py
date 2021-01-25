from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QLineEdit, QMessageBox
from spectramanipulator.dialogs.genericinputdialog import GenericInputDialog
from PyQt5.QtCore import Qt


class ParamSettingsDialog(GenericInputDialog):

    def __init__(self, exp_dep_params: list, equal_pars: dict = None, set_result=None):

        if self.is_opened:
            return

        self.exp_dep_params = exp_dep_params
        self.equal_pars = equal_pars

        widget_list = []
        self.lineedit_list = []

        for par_dict in self.exp_dep_params:
            le = QLineEdit(', '.join(par_dict['values']))
            widget_list.append((f"{par_dict['name']}:", le))
            self.lineedit_list.append(le)

        widget_list.append(('Set equal parameter pairs (enter groups of experiment indexes encapsulated in round brackets, use comma as delimiter):', None))
        self.pars_line_edits = []

        for i, par_name in enumerate(map(lambda d: d['name'], self.exp_dep_params)):
            text = ''
            if par_name in self.equal_pars:
                text = ', '.join([f"({', '.join(str(idx) for idx in pair)})" for pair in self.equal_pars[par_name]])

            le = QLineEdit(text)
            self.pars_line_edits.append(le)
            widget_list.append((f'{par_name}:', le))

        super(ParamSettingsDialog, self).__init__(widget_list=widget_list,
                                                  label_text='Set all experiment parameters, separate values by comma:',
                                                  title='Settings of experiment-dependent parameters',
                                                  parent=None,
                                                  set_result=set_result,
                                                  flags=Qt.WindowStaysOnTopHint)

    def accept(self):
        self.equal_pars = {}
        for par_dict, le, le_equal_par in zip(self.exp_dep_params, self.lineedit_list, self.pars_line_edits):
            split = list(filter(None, le.text().split(',')))
            n_exp = len(par_dict['values'])
            if len(split) != n_exp:
                QMessageBox.critical(self, "Error", "The number of values must match the number of experiments!", QMessageBox.Ok)
                return
            try:
                par_dict['values'] = list(map(lambda s: float(s.strip()), split))
            except ValueError:
                QMessageBox.critical(self, "Error", "The values cannot be parsed as float numbers!", QMessageBox.Ok)
                return

            par_name = par_dict['name']
            text = le_equal_par.text()

            # parse input pairs into list of tuples
            try:
                string = ''.join(filter(lambda d: not d.isspace(), list(text)))  # remove white space chars
                if string == '':
                    continue

                split_text = list(filter(None, string.split('),')))  # remove empty entries
                self.equal_pars[par_name] = []

                for pairs in split_text:
                    pairs = pairs.replace('(', '').replace(')', '')  # remove residual brackets
                    split = list(filter(None, pairs.split(',')))
                    idxs = []
                    for val in split:
                        int_val = int(val)
                        idxs.append(int_val)
                        if int_val >= n_exp or int_val < 0:  # index is < 0 or > num of experiments
                            raise ValueError(f"Index must be in range of (0, {n_exp}).")
                        # index must not repeat in pairs
                        if int_val in [idx for pair in self.equal_pars[par_name] for idx in pair]:
                            raise ValueError("Index must not repeat in pairs.")

                    self.equal_pars[par_name].append(idxs)

            except ValueError as e:
                QMessageBox.critical(self, "Error", e.__str__(), QMessageBox.Ok)
                return
            except:
                QMessageBox.critical(self, "Error", "The equal parameters are input incorrectly!", QMessageBox.Ok)
                return

        super(ParamSettingsDialog, self).accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = ParamSettingsDialog()
    Dialog.show()
    sys.exit(app.exec_())

