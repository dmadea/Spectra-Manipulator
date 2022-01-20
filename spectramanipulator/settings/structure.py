
import pyqtgraph as pg
from spectramanipulator.config_sel.groupcti import GroupCti
from spectramanipulator.config_sel.intcti import IntCti
from spectramanipulator.config_sel.floatcti import FloatCti, SnFloatCti
from spectramanipulator.config_sel.stringcti import StringCti
from spectramanipulator.config_sel.boolcti import BoolCti, BoolGroupCti
from spectramanipulator.config_sel.abstractcti import AbstractCti
from spectramanipulator.config_sel.choicecti import ChoiceCti

from spectramanipulator.config_sel.qtctis import ColorCti, FontCti, PenCti

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
        'description': 'Set the delimiter that separates the columns in the input file.'
    }


# groups with nothing has no value and no default value,
settings = {
    'items': [
        {
            'type': GroupCti,
            'name': 'Import',
            'description': 'Import settings.',
            'items': [
                {
                    'type': GroupCti,
                    'name': 'Parser',
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
                            'description': 'First n columns in input file will be skipped.',
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
                    'description': 'File import specific settings.',
                    'items': [
                        {
                            'type': GroupCti,
                            'name': 'DX file',
                            'description': 'DX file specific settings.',
                            'items': [
                                delimiter_setting_factory('Space'),
                                dec_separator_setting_factory(),
                                {
                                    'type': BoolCti,
                                    'name': 'If ##TITLE is empty...',
                                    'value': None,
                                    'default_value': True,
                                    'childrenDisabledValue': True,
                                    'description': 'If ##TITLE field is empty, set spectra name to filename.',
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
                            'description': 'CSV and other files specific settings.',
                            'items': [
                                {
                                    'type': BoolCti,
                                    'name': 'If header is empty...',
                                    'value': None,
                                    'default_value': True,
                                    'childrenDisabledValue': True,
                                    'description': 'If header field is empty, set spectra name to filename.',
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
                                    'description': 'CSV file specific settings.',
                                    'items': [
                                        delimiter_setting_factory('Comma'),
                                        dec_separator_setting_factory()
                                    ]
                                },
                                {
                                    'type': GroupCti,
                                    'name': 'Other files',
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