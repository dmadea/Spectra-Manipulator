from PyQt5 import QtWidgets

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QDialogButtonBox, QPushButton
from spectramanipulator.singleton import PersistentDialog

from spectramanipulator.configtree.configtreemodel import ConfigTreeModel
from spectramanipulator.configtree.configtreeview import ConfigTreeView
from spectramanipulator.configtree.groupcti import RootCti
from spectramanipulator.configtree.abstractcti import AbstractCti
from spectramanipulator.configtree.pgctis import PgColorMapCti

from spectramanipulator.settings.settings import Settings
from copy import deepcopy
from typing import Callable


class SettingsDialog(PersistentDialog):

    # flags=Qt.WindowStaysOnTopHint

    def __init__(self, accepted_func: Callable, rejected_func: Callable, applied_func: Callable,
                 parent=None, title='Settings Dialog', flags=Qt.WindowStaysOnTopHint):
        super(SettingsDialog, self).__init__(parent)

        self.accepted_func = accepted_func
        self.rejected_func = rejected_func
        self.applied_func = applied_func

        self.setWindowTitle(title)
        self.resize(500, 600)

        self.button_box = QDialogButtonBox(self)
        self.button_box.setOrientation(Qt.Horizontal)
        self.button_box.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok | QDialogButtonBox.Apply)

        self.button_box.accepted.connect(self.accept)  # OK button
        self.button_box.rejected.connect(self.reject)  # Cancel button
        apply_button = self.button_box.buttons()[-1]
        apply_button.pressed.connect(self.apply)

        self.restore_sett_button = QPushButton('Restore Default Settings')
        self.restore_sett_button.pressed.connect(self.restore_settings)

        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addWidget(self.restore_sett_button)
        self.bottom_layout.addStretch(0)
        self.bottom_layout.addWidget(self.button_box)

        self.VLayout = QVBoxLayout()

        self.tree_model = ConfigTreeModel()
        self.tree_view = ConfigTreeView(self.tree_model)

        self.sett = Settings()
        self.root_cti = RootCti(self.sett.find_node_by_xpath('/Public settings'), root_xpath='/Public settings')
        self.tree_model.setInvisibleRootItem(self.root_cti)
        self.tree_view.expandBranch()

        self.last_setting_dict = deepcopy(self.sett.settings)  # instantiate last settings

        def value_changed(cti: AbstractCti):
            data = cti.data
            if isinstance(cti, PgColorMapCti):
                data = data.key

            # print(f"value of {cti} has changed, new value: {data}")
            self.sett[cti.nodePath] = data

        self.tree_model.value_changed.connect(value_changed)

        self.VLayout.addWidget(self.tree_view)
        self.VLayout.addLayout(self.bottom_layout)

        self.setLayout(self.VLayout)

    def apply(self):
        self.applied_func()

    def accept(self) -> None:
        self.accepted_func()
        super(SettingsDialog, self).accept()

    def restore_settings(self):
        # maybe not necessary
        self.root_cti.resetToDefault(resetChildren=True)
        # TODO redraw Tree View
        # self.tree_model.sigItemChanged.emit(self.root_cti)
        # self.tree_model.dataChanged.emit()

    def reject(self):
        # keep the original settings
        self.sett.settings = self.last_setting_dict
        self.rejected_func()
        super(SettingsDialog, self).reject()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    dialog = SettingsDialog(lambda: None, lambda: None, lambda: None)

    dialog.show()

    sys.exit(app.exec_())


