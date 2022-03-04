from PyQt5 import QtWidgets

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QDialogButtonBox, QPushButton
from spectramanipulator.singleton import PersistentDialog

from spectramanipulator.configtree.configtreemodel import ConfigTreeModel
from spectramanipulator.configtree.configtreeview import ConfigTreeView
from spectramanipulator.configtree.groupcti import MainGroupCti
from spectramanipulator.configtree.abstractcti import AbstractCti
from spectramanipulator.configtree.pgctis import PgColorMapCti

from spectramanipulator.settings.settings import Settings
from copy import deepcopy
from typing import Callable


class RootCti(MainGroupCti):
    """ Configuration tree item for a main settings.
    """
    def __init__(self, nodeName='root'):
        super(RootCti, self).__init__(nodeName, root_xpath='/Public settings')

        def _insert_child(node: dict, parent_item: AbstractCti):
            kwargs = {key: value for key, value in node.items() if
                      key not in ['name', 'type', 'items', 'value', 'default_value']}

            # nodeName, defaultData
            cls = node['type']
            value = node['value'] if 'value' in node else None
            default_value = node['default_value'] if 'default_value' in node else None
            cti: AbstractCti = cls(node['name'], default_value, **kwargs)
            if value is not None:
                cti.data = node['value']
            parent_item.insertChild(cti)
            return cti

        def _insert_items(node_dict: dict, parent_item: AbstractCti):
            """Recursively inserts all items from settings"""
            if 'items' not in node_dict or len(node_dict['items']) == 0:
                return

            for dict_item in node_dict['items']:
                group_item = _insert_child(dict_item, parent_item)
                _insert_items(dict_item, group_item)

        _insert_items(Settings().find_node_by_xpath('/Public settings'), self)


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

        self.root_cti = RootCti()
        self.tree_model.setInvisibleRootItem(self.root_cti)
        self.tree_view.expandBranch()

        self.sett = Settings()
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


