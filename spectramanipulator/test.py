
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets
from config_sel.configtreemodel import ConfigTreeModel
from config_sel.configtreeview import ConfigTreeView
from config_sel.groupcti import MainGroupCti, GroupCti
from config_sel.intcti import IntCti
from config_sel.floatcti import FloatCti
from config_sel.stringcti import StringCti
from config_sel.boolcti import BoolCti, BoolGroupCti
from config_sel.abstractcti import AbstractCti

from config_sel.qtctis import ColorCti, FontCti, PenCti

from singleton import Singleton
from typing import Union

from PyQt5.QtGui import QPen

# node_structure = {
#     'type': class type,
#     'name': 'plotting',
#     'value': None,
#     'default_value': None,
#     'description': 'Long description of this setting...',
#     'items': [
#               {
#               ... same
#                },
#     ]
#     ...: ... and additional attributes for that CTI
# }

default_pen = pg.mkPen(color='red', width=1, style=1)


class Settings(Singleton):

    settings = {
        'items': [
            {
                'type': GroupCti,
                'name': 'Plotting',
                'value': None,
                'default_value': None,
                'description': 'Long description of this setting...',
                'items': [
                    {
                        'type': IntCti,
                        'name': 'Abc int',
                        'value': None,
                        'default_value': 5,
                        'minValue': 0,
                        'maxValue': 100,
                        'stepSize': 1,
                        'description': 'Long description of this setting...',
                    },
                    {
                        'type': StringCti,
                        'name': 'Random string',
                        'value': None,
                        'default_value': 'default',
                        'description': 'Long description of this setting...',
                    },
                    {
                        'type': ColorCti,
                        'name': 'Line color',
                        'value': None,
                        'default_value': 'blue',
                        'description': 'Long description of this setting...',
                    },
                    {
                        'type': PenCti,
                        'name': 'Pen',
                        'value': None,
                        'default_value': True,
                        'resetTo': default_pen,
                        'description': 'Long description of this setting...',
                    },
                    {
                        'type': GroupCti,
                        'name': 'X axis',
                        'value': None,
                        'default_value': None,
                        'description': 'Long description of this setting...',
                        'items': [
                            {
                                'type': StringCti,
                                'name': 'Label',
                                'value': 'some val',
                                'default_value': 'Wavelength / nm',
                                'description': 'Setting...',
                            },
                            {
                                'type': IntCti,
                                'name': 'Font size',
                                'value': None,
                                'default_value': 10,
                                'minValue': 0,
                                'maxValue': 100,
                                'stepSize': 1,
                                'description': 'Long description of this setting...',
                            },
                        ]
                    },
                ]
            },

        ]
    }

    def __init__(self):
        pass

    def iterate_settings(self, node=None, xpath=''):
        """
        Recursive generator to iterate all the settings. Returns tuple of xpath and actual node.
        """
        node = self.settings if node is None else node

        if 'items' not in node or len(node['items']) == 0:
            return

        for item in node['items']:
            new_xpath = f"{xpath}.{item['name']}"
            yield new_xpath, item
            yield from self.iterate_settings(item, xpath=new_xpath)

    def _find_node_by_xpath(self, path):
        # split path by . and find return the value
        splitted = filter(None, path.split('.'))  # remove empty entries

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


class RootCti(MainGroupCti):
    """ Configuration tree item for a PgImagePlot2dCti inspector
    """
    def __init__(self, nodeName='root'):
        """ Constructor

            Maintains a link to the target pgImagePlot2d inspector, so that changes in the
            configuration can be applied to the target by simply calling the apply method.
            Vice versa, it can connect signals to the target.
        """
        super(RootCti, self).__init__(nodeName)

        def _insert_child(node: dict, parent_item: AbstractCti):
            kwargs = {key: value for key, value in node.items() if
                      key not in ['name', 'type', 'items', 'value', 'default_value']}

            # nodeName, defaultData
            cls = node['type']
            cti: AbstractCti = cls(node['name'], node['default_value'], **kwargs)
            if node['value'] is not None:
                cti.data = node['value']
            parent_item.insertChild(cti)
            return cti

        def _insert_items(node_dict: Union[None, dict], parent_item: AbstractCti):
            """Recursively inserts all items from settings"""
            if 'items' not in node_dict or len(node_dict['items']) == 0:
                return

            for dict_item in node_dict['items']:
                group_item = _insert_child(dict_item, parent_item)
                _insert_items(dict_item, group_item)

        _insert_items(Settings().settings, self)


if __name__ == '__main__':

    s = Settings()  # singleton settings

    # for xpath, item in s.iterate_settings():
    #     print(xpath)

    # print(s['.plotting.abc int'])
    # print(s['x axis.label'])
    # s['.x axis..label'] = 'some different label'
    # print(s['x axis.label'])
    #
    #
    #
    # a = 1

    app = QtWidgets.QApplication(sys.argv)

    tm = ConfigTreeModel()
    tv = ConfigTreeView(tm)

    mg_cti = RootCti('main')
    tm.setInvisibleRootItem(mg_cti)
    tv.expandBranch()

    tv.show()
    sys.exit(app.exec())

