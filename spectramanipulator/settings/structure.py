
import pyqtgraph as pg
from spectramanipulator.configtree.groupcti import GroupCti
from spectramanipulator.configtree.intcti import IntCti
from spectramanipulator.configtree.floatcti import FloatCti, SnFloatCti
from spectramanipulator.configtree.stringcti import StringCti
from spectramanipulator.configtree.boolcti import BoolCti, BoolGroupCti
from spectramanipulator.configtree.pgctis import PgColorMapCti
from spectramanipulator.configtree.choicecti import ChoiceCti
from spectramanipulator.configtree.qtctis import ColorCti, PenCti
from PyQt5.QtCore import Qt

from spectramanipulator.configtree.colors import DEFAULT_COLOR_MAP


# from spectramanipulator.configtree.qtctis import ColorCti, FontCti, PenCti

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


def get_delimiter_from_idx(idx: int):
    return list(delimiter_separators.values())[idx]


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
public_settings = [
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
                        'name': 'Clipboard',
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
        ]
    },
    {
        'type': GroupCti,
        'name': 'Export',
        'description': 'Export settings.',
        'items': [
            {
                'type': GroupCti,
                'name': 'Files',
                'description': 'Files export settings.',
                'items': [
                    {
                        'type': BoolCti,
                        'name': 'Include group name',
                        'value': None,
                        'default_value': False,
                        'description': 'Includes the group name before the actual data when exporting to file.',
                    },
                    {
                        'type': BoolCti,
                        'name': 'Include header',
                        'value': None,
                        'default_value': True,
                        'description': 'Includes the header row (x axis name and name(s) of the spectra) before'
                                       ' the actual data when exporting to file.',
                    },
                ]

            },
            {
                'type': GroupCti,
                'name': 'Clipboard',
                'description': 'Clipboard export settings. Sets, how to export text data to clipboard (in order'
                               'to retain compatibility with MS Excel/Origin, keep delimiter set as tabulator \\t,'
                               'decimal separator is application specific).',
                'items': [
                    {
                        'type': BoolCti,
                        'name': 'Include group name',
                        'value': None,
                        'default_value': False,
                        'description': 'Includes the group name before the actual data when exporting to clipboard.',
                    },
                    {
                        'type': BoolCti,
                        'name': 'Include header',
                        'value': None,
                        'default_value': True,
                        'description': 'Includes the header row (x axis name and name(s) of the spectra) before'
                                       ' the actual data when exporting to clipboard.',
                    },
                    delimiter_setting_factory('Horizontal tabulator \\t'),
                    dec_separator_setting_factory()
                ]

            },
        ]

    },
    {
        'type': GroupCti,
        'name': 'Plotting',
        'description': 'Plotting settings.',
        'items': [
            {
                'type': GroupCti,
                'name': 'Color and line style',
                'description': 'Color and line style settings.',
                'items': [
                    {
                        'type': BoolCti,
                        'name': 'Plot spectra with same color in groups',
                        'value': None,
                        'default_value': False,
                        'description': 'If True, the spectra in groups will be plotted with the same color.',
                    },
                    {
                        'type': BoolCti,
                        'name': 'Plot spectra with different line style among groups',
                        'value': None,
                        'default_value': False,
                        'description': 'If True, the line style of the spectra will be different for different groups.',
                    },
                    {
                        'type': BoolCti,
                        'name': 'Reversed Z order',
                        'value': None,
                        'default_value': False,
                        'description': 'If True, first spectra will be plotted behind the next ones.',
                    },
                    {
                        'type': SnFloatCti,
                        'name': 'Line width',
                        'value': None,
                        'default_value': 1.0,
                        'minValue': 0.1,
                        'maxValue': 100.0,
                        'precision': 2,
                        'description': 'Line width of the plotted spectra.',
                    },
                    {
                        'type': BoolCti,
                        'name': 'Use gradient colormap',
                        'value': None,
                        'default_value': False,
                        'childrenDisabledValue': False,
                        'description': 'If True, gradient colormap will be used, otherwise, default colors '
                                       '(red, green, blue, black, yellow, magenta, cyan, gray, ... and repeat) will be used',
                        'items': [
                            {
                                'type': PgColorMapCti,
                                'name': 'Colormap',
                                'value': None,
                                'default_value': DEFAULT_COLOR_MAP,
                                'items': [
                                    {
                                        'type': BoolCti,
                                        'name': 'Automatically spread',
                                        'value': None,
                                        'default_value': True,
                                        'childrenDisabledValue': True,
                                        'description': 'No matter what number of spectra are plotted, if true, colormap will automatically spread over all spectra.',
                                        'items': [
                                            {
                                                'type': IntCti,
                                                'name': 'Number of spectra',
                                                'value': None,
                                                'default_value': 10,
                                                'minValue': 1,
                                                'maxValue': 9999999,
                                                'stepSize': 1,
                                                'description': 'Number of spectra plotted with the selected gradient.',
                                            },
                                        ]
                                    },
                                    {
                                        'type': SnFloatCti,
                                        'name': 'Start range',
                                        'value': None,
                                        'default_value': 0.0,
                                        'minValue': 0.0,
                                        'maxValue': 1.0,
                                        'precision': 2,
                                        'description': 'Start range of the colormap used (value from 0 to 1), default 0.',
                                    },
                                    {
                                        'type': SnFloatCti,
                                        'name': 'End range',
                                        'value': None,
                                        'default_value': 1.0,
                                        'minValue': 0.0,
                                        'maxValue': 1.0,
                                        'precision': 2,
                                        'description': 'End range of the colormap used (value from 0 to 1), default 1.',
                                    },
                                    {
                                        'type': BoolCti,
                                        'name': 'Reversed',
                                        'value': None,
                                        'default_value': False,
                                        'description': 'If True, the colormap will be reversed.',
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
            {
                'type': GroupCti,
                'name': 'Graph',
                'description': 'Graph settings.',
                'items': [
                    {
                        'type': BoolCti,
                        'name': 'Antialiasing',
                        'value': None,
                        'default_value': True,
                    },
                    {
                        'type': StringCti,
                        'name': 'Graph title',
                        'value': None,
                        'default_value': '',
                    },
                    {
                        'type': StringCti,
                        'name': 'X axis label',
                        'value': None,
                        'default_value': 'Wavelength / nm',
                    },
                    {
                        'type': StringCti,
                        'name': 'Y axis label',
                        'value': None,
                        'default_value': 'Absorbance',
                    },

                    {
                        'type': SnFloatCti,
                        'name': 'Legend spacing',
                        'value': None,
                        'default_value': 8.0,
                        'minValue': 0.0,
                        'maxValue': 200,
                        'precision': 1,
                        'description': 'Spacing of the legend labels.',
                    },
                ]
            },


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
    }
]

line_types = [
    {'name': 'Solid line', 'index': Qt.SolidLine},
    {'name': 'Dashed line', 'index': Qt.DashLine},
    {'name': 'Dotted line', 'index': Qt.DotLine},
    {'name': 'Dash-dotted line', 'index': Qt.DashDotLine},
    {'name': 'Dash-dot-dotted line', 'index': Qt.DashDotDotLine},
    {'name': 'No line', 'index': Qt.NoPen}
]

symbol_types = [
    {'name': 'No symbol', 'sym': None},
    {'name': '\u25CF', 'sym': 'o'},
    {'name': '\u25BC', 'sym': 't'},
    {'name': '\u25B2', 'sym': 't1'},
    {'name': '\u25BA', 'sym': 't2'},
    {'name': '\u25C4', 'sym': 't3'},
    {'name': '\u25A0', 'sym': 's'},
    {'name': '\u2B1F', 'sym': 'p'},
    {'name': '\u2B22', 'sym': 'h'},
    {'name': '\u2605', 'sym': 'star'},
    {'name': '+', 'sym': '+'},
    {'name': '\u2666', 'sym': 'd'}
]


style_settings = [
    {
        'type': BoolCti,
        'name': 'Line color',
        'value': None,
        'default_value': False,
        'childrenDisabledValue': True,
        'description': 'Skips columns that contains the NaN (Not a Number) values.',
        'items': [
            {
                'type': ColorCti,
                'name': 'Line color',
                'value': None,
                'default_value': 'blue',
                'description': 'Line color',
            },
            {
                'type': IntCti,
                'name': 'Alpha',
                'value': None,
                'default_value': 255,
                'minValue': 0,
                'maxValue': 255,
                'description': 'Value that replaces NaN values.',
            }
        ]
    },
    {
        'type': BoolCti,
        'name': 'Line style',
        'value': None,
        'default_value': False,
        'childrenDisabledValue': True,
        'description': 'Skips columns that contains the NaN (Not a Number) values.',
        'items': [
            {
                'type': SnFloatCti,
                'name': 'Line width',
                'value': None,
                'default_value': 1,
                'precision': 1,
                'minValue': 0,
                'maxValue': 100,
                'description': 'Value that replaces NaN values.',
            },
            {
                'type': ChoiceCti,
                'name': 'Line type',
                'value': None,
                'default_value': 1,
                'configValues': list(map(lambda d: d['name'], line_types)),
                'description': '...',  # TODO
            }
        ]
    },
    {
        'type': BoolCti,
        'name': 'Symbol style',
        'value': None,
        'default_value': False,
        'childrenDisabledValue': True,
        'description': 'Skips columns that contains the NaN (Not a Number) values.',
        'items': [
            {
                'type': SnFloatCti,
                'name': 'Size',
                'value': None,
                'default_value': 1,
                'precision': 1,
                'minValue': 0,
                'maxValue': 100,
                'description': 'Value that replaces NaN values.',
            },
            {
                'type': ChoiceCti,
                'name': 'Type',
                'value': None,
                'default_value': 1,
                'configValues': list(map(lambda d: d['name'], symbol_types)),
                'description': '...',  # TODO
            },
            {
                'type': SnFloatCti,
                'name': 'Line type',
                'value': None,
                'default_value': 1,
                'precision': 1,
                'minValue': 0,
                'maxValue': 100,
                'description': 'Value that replaces NaN values.',
            },
            {
                'type': BoolCti,
                'name': 'Brush color',
                'value': None,
                'default_value': False,
                'childrenDisabledValue': True,
                'description': 'Skips columns that contains the NaN (Not a Number) values.',
                'items': [
                    {
                        'type': ColorCti,
                        'name': 'Brush color',
                        'value': None,
                        'default_value': 'blue',
                        'description': 'Line color',
                    },
                    {
                        'type': IntCti,
                        'name': 'Alpha',
                        'value': None,
                        'default_value': 255,
                        'minValue': 0,
                        'maxValue': 255,
                        'description': 'Value that replaces NaN values.',
                    }
                ]
            },
            {
                'type': BoolCti,
                'name': 'Fill color',
                'value': None,
                'default_value': False,
                'childrenDisabledValue': True,
                'description': 'Skips columns that contains the NaN (Not a Number) values.',
                'items': [
                    {
                        'type': ColorCti,
                        'name': 'Fill color',
                        'value': None,
                        'default_value': 'blue',
                        'description': 'Line color',
                    },
                    {
                        'type': IntCti,
                        'name': 'Alpha',
                        'value': None,
                        'default_value': 255,
                        'minValue': 0,
                        'maxValue': 255,
                        'description': 'Value that replaces NaN values.',
                    }
                ]
            },
        ]
    },
    {
        'type': BoolCti,
        'name': 'Plot legend',
        'value': None,
        'default_value': True,
        'description': 'If False, legend will not be shown for the selected items.',
    }

]