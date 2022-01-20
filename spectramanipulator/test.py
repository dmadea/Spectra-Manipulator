
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets
from spectramanipulator.config_sel.configtreemodel import ConfigTreeModel
from spectramanipulator.config_sel.configtreeview import ConfigTreeView
from spectramanipulator.config_sel.groupcti import MainGroupCti
from spectramanipulator.config_sel.abstractcti import AbstractCti
from spectramanipulator.settings.structure import settings

from spectramanipulator.singleton import Singleton
from typing import Union

from PyQt5.QtGui import QPen


class Settings(Singleton):

    def __init__(self):

        self.settings = settings

    def iterate_settings(self, node=None, xpath=''):
        """
        Recursive generator to iterate all the settings. Returns tuple of xpath and actual node.
        """
        node = self.settings if node is None else node

        if 'items' not in node or len(node['items']) == 0:
            return

        for item in node['items']:
            new_xpath = f"{xpath}/{item['name']}"
            yield new_xpath, item
            yield from self.iterate_settings(item, xpath=new_xpath)

    def _find_node_by_xpath(self, path):
        # split path by / and find return the value
        splitted = filter(None, path.split('/'))  # remove empty entries

        current_node = self.settings

        for name in splitted:
            sel_nodes = list(filter(lambda node: node['name'] == name, current_node['items']))
            if len(sel_nodes) == 0:
                raise ValueError(f"Setting '{path} could not be found.")
            current_node = sel_nodes[0]

        return current_node

    def __getitem__(self, key: str):
        node = self._find_node_by_xpath(key)
        return node['value']

    def __setitem__(self, key, value):
        node = self._find_node_by_xpath(key)
        node['value'] = value


# class RootCti(MainGroupCti):
#     """ Configuration tree item for a main settings.
#     """
#     def __init__(self, nodeName='root'):
#         super(RootCti, self).__init__(nodeName)
#
#         def _insert_child(node: dict, parent_item: AbstractCti):
#             kwargs = {key: value for key, value in node.items() if
#                       key not in ['name', 'type', 'items', 'value', 'default_value']}
#
#             # nodeName, defaultData
#             cls = node['type']
#             value = node['value'] if 'value' in node else None
#             default_value = node['default_value'] if 'default_value' in node else None
#             cti: AbstractCti = cls(node['name'], default_value, **kwargs)
#             if value is not None:
#                 cti.data = node['value']
#             parent_item.insertChild(cti)
#             return cti
#
#         def _insert_items(node_dict: Union[None, dict], parent_item: AbstractCti):
#             """Recursively inserts all items from settings"""
#             if 'items' not in node_dict or len(node_dict['items']) == 0:
#                 return
#
#             for dict_item in node_dict['items']:
#                 group_item = _insert_child(dict_item, parent_item)
#                 _insert_items(dict_item, group_item)
#
#         _insert_items(Settings().settings, self)


if __name__ == '__main__':

    s = Settings()  # singleton settings

    for xpath, item in s.iterate_settings():
        print(xpath)

    # print(s['.plotting.abc int'])
    # print(s['x axis.label'])
    # s['.x axis..label'] = 'some different label'
    # print(s['x axis.label'])
    # a = 1

    app = QtWidgets.QApplication(sys.argv)

    tm = ConfigTreeModel()
    tv = ConfigTreeView(tm)

    mg_cti = RootCti()
    tm.setInvisibleRootItem(mg_cti)
    tv.expandBranch()

    def value_changed(cti: AbstractCti):
        print("value has changed", cti)
        s[cti.nodePath] = cti.data

    tm.value_changed.connect(value_changed)

    tv.show()
    sys.exit(app.exec())

