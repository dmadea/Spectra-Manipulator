from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QLineEdit, QMessageBox
from spectramanipulator.dialogs.genericinputdialog import GenericInputDialog


class ParamSettingsDialog(GenericInputDialog):

    def __init__(self, exp_dep_params: list, set_result=None):

        if self.is_opened:
            return

        self.exp_dep_params = exp_dep_params

        widget_list = []
        self.lineedit_list = []

        for par_dict in self.exp_dep_params:
            le = QLineEdit(', '.join(par_dict['values']))
            widget_list.append((f"{par_dict['name']}:", le))
            self.lineedit_list.append(le)

        super(ParamSettingsDialog, self).__init__(widget_list=widget_list,
                                                  label_text='Set all experiment parameters, separate values by comma:',
                                                  title='Settings of experiment-dependent parameters',
                                                  parent=None,
                                                  set_result=set_result)

    def accept(self):
        for par_dict, le in zip(self.exp_dep_params, self.lineedit_list):
            split = list(filter(None, le.text().split(',')))
            if len(split) != len(par_dict['values']):
                QMessageBox.critical(self, "Error", "The number of values must match the number of experiments!", QMessageBox.Ok)
                return
            try:
                par_dict['values'] = list(map(lambda s: float(s.strip()), split))
            except ValueError:
                QMessageBox.critical(self, "Error", "The values cannot be parsed as float numbers!", QMessageBox.Ok)
                return

        super(ParamSettingsDialog, self).accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = ParamSettingsDialog()
    Dialog.show()
    sys.exit(app.exec_())

