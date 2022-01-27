import logging
# import sys
# import pyqtgraph as pg
# from PyQt5 import QtWidgets
# from spectramanipulator.config_sel.configtreemodel import ConfigTreeModel
# from spectramanipulator.config_sel.configtreeview import ConfigTreeView
# from spectramanipulator.config_sel.abstractcti import AbstractCti
from spectramanipulator.settings.structure import public_settings

from spectramanipulator.singleton import Singleton
import os
import json

from PyQt5.QtGui import QPen


class Settings(Singleton):

    _config_filename = "config_new.json"

    enable_multiprocessing = True  # enables importing files as separate processes
    force_multiprocess = False  # importing of files will only run in multiple processes

    general_models_dir = 'general models'
    REG_PROGRAM_NAME = 'SpectraManipulator.projectfile'
    PROJECT_EXTENSION = '.smpj'

    def __init__(self):

        self.settings = {
            'items': [
                {
                    'name': 'Public settings',
                    'items': public_settings
                },
                {
                    'name': 'Private settings',
                    'items': [
                        dict(name='Import files dialog path', value=""),
                        dict(name='Import LPF dialog path', value=""),
                        dict(name='Import EEM dialog path', value=""),
                        dict(name='Open project dialog path', value=""),
                        dict(name='Save project dialog path', value=""),
                        dict(name='Load kinetics last path', value=""),
                        dict(name='Recent project filepaths', value=[]),
                        {
                            'name': 'Export spectra as dialog',
                            'items': [
                                {
                                    'name': 'Path',
                                    'value': ""
                                },
                                {
                                    'name': 'Ext',
                                    'value': ".txt"
                                },
                                {
                                    'name': 'Delimiter',
                                    'value': "\t"
                                },
                                {
                                    'name': 'Decimal separator',
                                    'value': "."
                                },
                            ]
                        },
                    ]
                },
            ]
        }

    def iterate_settings(self, only_with_value=True):
        """
        Recursive generator to iterate all the settings. Returns tuple of xpath and actual node.
        """
        def _iter_sett(node: dict, xpath: str):
            if 'items' not in node or len(node['items']) == 0:
                return

            for item in node['items']:
                new_xpath = f"{xpath}/{item['name']}"
                if only_with_value and 'value' in item:
                    yield new_xpath, item

                if not only_with_value:
                    yield new_xpath, item
                yield from _iter_sett(item, xpath=new_xpath)

        return _iter_sett(self.settings, '')

    def find_node_by_xpath(self, path):
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
        node = self.find_node_by_xpath(key)
        return node['value']

    def __setitem__(self, key, value):
        node = self.find_node_by_xpath(key)
        node['value'] = value

    @classmethod
    def get_config_filepath(cls):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(curr_dir, cls._config_filename)

    def save(self):
        sett = {}
        for xpath, item in self.iterate_settings(True):
            value = None
            if 'value' in item:
                value = item['value']
            if value is None and 'default_value' in item:
                value = item['default_value']
            sett[xpath] = value

        try:
            with open(Settings.get_config_filepath(), "w") as file:
                json.dump(sett, file, sort_keys=False, indent=4, separators=(',', ': '))

        except Exception as ex:
            logging.error("Error saving settings to file {}. Error message:\n{}".format(
                Settings._config_filename, ex.__str__()))

    def load(self):
        try:
            with open(Settings.get_config_filepath(), "r") as file:
                data = json.load(file)

            for key, value in data.items():
                self[key] = value

        except Exception as ex:
            logging.error(
                "Error loading settings from file {}, setting up default settings. Error message:\n{}".format(
                    Settings._config_filename, ex.__str__()))


if __name__ == '__main__':

    s = Settings()  # singleton settings

    # s.save()
    s.load()

    for xpath, item in s.iterate_settings():
        print(xpath, item['value'])

    # print(s['.plotting.abc int'])
    # print(s['x axis.label'])
    # s['.x axis..label'] = 'some different label'
    # print(s['x axis.label'])
    # a = 1
    #
    # app = QtWidgets.QApplication(sys.argv)
    #
    # tm = ConfigTreeModel()
    # tv = ConfigTreeView(tm)
    #
    # mg_cti = RootCti()
    # tm.setInvisibleRootItem(mg_cti)
    # tv.expandBranch()
    #
    # def value_changed(cti: AbstractCti):
    #     print("value has changed", cti)
    #     s[cti.nodePath] = cti.data
    #
    # tm.value_changed.connect(value_changed)
    #
    # tv.show()
    # sys.exit(app.exec())

