
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets
from config_sel.configtreemodel import ConfigTreeModel
from config_sel.configtreeview import ConfigTreeView
from config_sel.groupcti import MainGroupCti, GroupCti
from config_sel.intcti import IntCti
from config_sel.floatcti import FloatCti, SnFloatCti
from config_sel.stringcti import StringCti
from config_sel.boolcti import BoolCti, BoolGroupCti
from config_sel.abstractcti import AbstractCti
from config_sel.choicecti import ChoiceCti

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

delimiter_separators = {
    'Any whitespace': '',
    'Horizontal tabulator \\t': '\t',
    'Comma': ',',
    'Dot': '.',
    'Space': ' ',
    'New Line \\n': '\n',
    'Carriage return \\r': '\r',
    'Vertical tabulator \\v': '\v',
    'Bell \\a': '\a',
    'Backspace \\b': '\b',
    'Formfeed \\f': '\f'
}


def dec_separator_setting_factory():
    return {
        'type': StringCti,
        'name': 'Decimal separator',
        'value': None,
        'default_value': '.',
        'description': 'Set the decimal separator of values in text.',
    }


def delimiter_setting_factory(default_value: str = 'Any whitespace'):
    configValues = list(delimiter_separators.keys())

    return {
        'type': ChoiceCti,
        'name': 'Delimiter',
        'value': None,
        'default_value': configValues.index(default_value),
        'configValues': configValues,
        'editable': False,
        'description': 'Set the delimiter that separates the columns',  # TODO
    }


class Settings(Singleton):

    settings = {
        'items': [
            {
                'type': GroupCti,
                'name': 'Import',
                'value': None,
                'default_value': None,
                'description': 'Import settings.',
                'items': [
                    {
                        'type': GroupCti,
                        'name': 'Parser',
                        'value': None,
                        'default_value': None,
                        'description': 'Input parser specific settings.',
                        'items': [
                            {
                                'type': BoolCti,
                                'name': 'Remove empty entries',
                                'value': None,
                                'default_value': True,
                                'description': 'Removes empty entries for a given delimiter for each parsed lines.',
                            },
                            {
                                'type': IntCti,
                                'name': 'Skip first',
                                'value': None,
                                'default_value': 0,
                                'minValue': 0,
                                'maxValue': 100000,
                                'stepSize': 1,
                                'suffix': ' columns',
                                'description': 'First n columns in input data will be skipped.',
                            },
                            {
                                'type': BoolCti,
                                'name': 'Skip columns containing NaNs',
                                'value': None,
                                'default_value': False,
                                'childrenDisabledValue': True,
                                'description': 'Skips columns that contains the NaN (Not a Number) values.',
                                'items': [
                                    {
                                        'type': SnFloatCti,
                                        'name': 'NaN value replacement',
                                        'value': None,
                                        'default_value': 0,
                                        'precision': 5,
                                        'description': 'Value that replaces NaN values.',
                                    }
                                ]
                            },
                            {
                                'type': GroupCti,
                                'name': 'Clipoboard',
                                'value': None,
                                'default_value': None,
                                'description': 'Import from clipboard settings.',
                                'items': [
                                    {
                                        'type': BoolCti,
                                        'name': 'Import as text from Excel',
                                        'value': None,
                                        'default_value': False,
                                        'description': 'Imports data from MS Excel as text (if checked, decimal precision will be lost).'
                                                       ' If unchecked, data are imported from XML data format where the numbers are'
                                                       ' stored in full precision.',
                                    },
                                    delimiter_setting_factory('Horizontal tabulator \\t'),
                                    dec_separator_setting_factory(),
                                ]
                            },
                        ]
                    },
                    {
                        'type': GroupCti,
                        'name': 'Files',
                        'value': None,
                        'default_value': None,
                        'description': 'File import specific settings.',
                        'items': [
                            {
                                'type': GroupCti,
                                'name': 'DX file',
                                'value': None,
                                'default_value': None,
                                'description': 'DX file specific settings.',
                                'items': [
                                    delimiter_setting_factory('Space'),
                                    dec_separator_setting_factory(),
                                    {
                                        'type': BoolCti,
                                        'name': 'If ##TITLE is empty, \nuse spectra name from filename',
                                        'value': None,
                                        'default_value': True,
                                        'childrenDisabledValue': True,
                                        'description': '...', # TODO
                                        'items': [
                                            {
                                                'type': ChoiceCti,
                                                'name': 'Import spectra name from',
                                                'value': None,
                                                'default_value': 1,
                                                'configValues': ['Filename', '##TITLE entry'],
                                                'description': '...', # TODO
                                            }
                                        ]
                                    },
                                ]
                            },
                            {
                                'type': GroupCti,
                                'name': 'CSV and other files',
                                'value': None,
                                'default_value': None,
                                'description': 'CSV and other files specific settings.',
                                'items': [
                                    {
                                        'type': BoolCti,
                                        'name': 'If header is empty, \nuse spectra name from filename',
                                        'value': None,
                                        'default_value': True,
                                        'childrenDisabledValue': True,
                                        'description': '...',  # TODO
                                        'items': [
                                            {
                                                'type': ChoiceCti,
                                                'name': 'Import spectra name from',
                                                'value': None,
                                                'default_value': 1,
                                                'configValues': ['Filename', 'Header'],
                                                'description': '...',  # TODO
                                            }
                                        ]
                                    },
                                    {
                                        'type': GroupCti,
                                        'name': 'CSV file',
                                        'value': None,
                                        'default_value': None,
                                        'description': 'CSV file specific settings.',
                                        'items': [
                                            delimiter_setting_factory('Comma'),
                                            dec_separator_setting_factory()
                                        ]
                                    },
                                    {
                                        'type': GroupCti,
                                        'name': 'Other files',
                                        'value': None,
                                        'default_value': None,
                                        'description': 'CSV file specific settings.',
                                        'items': [
                                            delimiter_setting_factory('Any whitespace'),
                                            dec_separator_setting_factory()
                                        ]
                                    },
                                ]
                            },

                        ]
                    },


                    # {
                    #     'type': BoolCti,
                    #     'name': 'bool cti test',
                    #     'value': None,
                    #     'default_value': True,
                    #     'childrenDisabledValue': False,
                    #     'description': 'Long description of this setting...',
                    #     'items': [
                    #         {
                    #             'type': StringCti,
                    #             'name': 'bool Random string 1',
                    #             'value': None,
                    #             'default_value': 'Random',
                    #             'description': 'Long description of this setting...',
                    #         },
                    #         {
                    #             'type': StringCti,
                    #             'name': 'bool Random string 2',
                    #             'value': None,
                    #             'default_value': '56sad65  4sd5',
                    #             'description': 'Long description of this setting...',
                    #         },
                    #     ]
                    # },
                    # {
                    #     'type': ColorCti,
                    #     'name': 'Line color',
                    #     'value': None,
                    #     'default_value': 'blue',
                    #     'description': 'Long description of this setting...',
                    # },
                    # {
                    #     'type': PenCti,
                    #     'name': 'Pen',
                    #     'value': None,
                    #     'default_value': True,
                    #     'resetTo': default_pen,
                    #     'description': 'Long description of this setting...',
                    # },
                    # {
                    #     'type': GroupCti,
                    #     'name': 'X axis',
                    #     'value': None,
                    #     'default_value': None,
                    #     'description': 'Long description of this setting...',
                    #     'items': [
                    #         {
                    #             'type': StringCti,
                    #             'name': 'Label',
                    #             'value': 'some val',
                    #             'default_value': 'Wavelength / nm',
                    #             'description': 'Setting...',
                    #         },
                    #         {
                    #             'type': IntCti,
                    #             'name': 'Font size',
                    #             'value': None,
                    #             'default_value': 10,
                    #             'minValue': 0,
                    #             'maxValue': 100,
                    #             'stepSize': 1,
                    #             'description': 'Long description of this setting...',
                    #         },
                    #     ]
                    # },
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
    """ Configuration tree item for a main settings.
    """
    def __init__(self, nodeName='root'):
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

